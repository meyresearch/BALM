import os
from typing import Union

import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.tracking import WandBTracker
from datasets import Dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

import wandb

from balm.configs import Configs
from balm.datasets import create_scaffold_split_dti
from balm.datasets.utils import DataCollatorWithPadding, get_dataset_split
from balm.models import BaselineModel, BALM
from balm.models.utils import load_trained_model, load_pretrained_pkd_bounds
from balm.metrics import get_ci, get_pearson, get_rmse, get_spearman
from balm.tokenization import pre_tokenize_unique_entities, tokenize_with_lookup
from balm import factories


class Trainer:
    def __init__(
        self, configs: Configs, wandb_entity: str, wandb_project: str, outputs_dir: str
    ):
        self.configs = configs

        self.gradient_accumulation_steps = (
            self.configs.model_configs.model_hyperparameters.gradient_accumulation_steps
        )
        self.protein_max_seq_len = (
            self.configs.model_configs.model_hyperparameters.protein_max_seq_len
        )
        self.drug_max_seq_len = (
            self.configs.model_configs.model_hyperparameters.drug_max_seq_len
        )

        self.outputs_dir = outputs_dir
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project

        self.protein_tokenizer, self.drug_tokenizer = self._load_tokenizers()

        if (
            self.configs.model_configs.protein_fine_tuning_type == "baseline"
            and self.configs.model_configs.drug_fine_tuning_type == "baseline"
        ):
            self.model = BaselineModel(self.configs.model_configs)
        else:
            self.model = BALM(self.configs.model_configs)

        self.train_dataloader = None
        self.valid_dataloader = None
        self.test_dataloader = None

        self._setup_run_name()

    def _load_tokenizers(self):
        protein_tokenizer = AutoTokenizer.from_pretrained(
            self.configs.model_configs.protein_model_name_or_path
        )
        drug_tokenizer = AutoTokenizer.from_pretrained(
            self.configs.model_configs.drug_model_name_or_path
        )

        return protein_tokenizer, drug_tokenizer

    def set_pkd_bounds(self, dataset):
        self.pkd_lower_bound = min(dataset.y)
        self.pkd_upper_bound = max(dataset.y)

        # If we are loading a trained model for a zero-shot experiment, load the pkd bounds from the training (at the moment, it's manually collected)
        if self.configs.model_configs.checkpoint_path:
            if self.configs.dataset_configs.train_ratio == 0.0:
                self.pkd_lower_bound, self.pkd_upper_bound = load_pretrained_pkd_bounds(self.configs.model_configs.checkpoint_path)
        
        print(
            f"Scaling labels: from {self.pkd_lower_bound} - {self.pkd_upper_bound} to -1 to 1"
        )

    def set_dataset(self, *args, **kwargs) -> dict:
        dataset = factories.get_dataset(self.configs.dataset_configs.dataset_name)

        print(
            f"Training with {self.configs.model_configs.loss_function} loss function."
        )

        if self.configs.model_configs.loss_function == "cosine_mse":
            self.set_pkd_bounds(dataset)

            if self.pkd_upper_bound == self.pkd_lower_bound:
                # To handle the hacky case where all labels are the same
                dataset.y = [0 for _ in dataset.y]
            else:
                dataset.y = [
                    (pkd - self.pkd_lower_bound)
                    / (self.pkd_upper_bound - self.pkd_lower_bound)
                    * 2
                    - 1
                    for pkd in dataset.y
                ]
            # Unique preprocessing for non-TDC dataset, preprocess Y column using the same pkd scaling
            if not self.configs.dataset_configs.dataset_name.startswith("DTI_"):
                if self.pkd_upper_bound == self.pkd_lower_bound:
                    # To handle the case where all labels are the same
                    dataset.data["Y"] = dataset.data["Y"].apply(lambda x: 0)
                else:
                    dataset.data["Y"] = dataset.data["Y"].apply(
                        lambda x: (x - self.pkd_lower_bound)
                        / (self.pkd_upper_bound - self.pkd_lower_bound)
                        * 2
                        - 1
                    )
        elif self.configs.model_configs.loss_function in ["baseline_mse"]:
            print("Using original pKd")

        print("Filtering dataset by length")
        print(f"Protein max length: {self.protein_max_seq_len}")
        print(f"Drug max length: {self.drug_max_seq_len}")

        dataset_splits = {}
        for split, data_df in get_dataset_split(dataset).items():
            if data_df is None:
                continue
            # Pre-tokenize unique ligands and proteins for this split
            print(f"Pre-tokenize unique ligands and proteins for {split}")
            protein_tokenized_dict, drug_tokenized_dict = pre_tokenize_unique_entities(
                data_df,
                self.protein_tokenizer,
                self.drug_tokenizer,
            )

            # Use the optimized tokenization for this split
            dataset = Dataset.from_pandas(data_df).map(
                lambda x: tokenize_with_lookup(
                    x, protein_tokenized_dict, drug_tokenized_dict
                ),
            )
            num_original_dataset = len(dataset)
            dataset = dataset.filter(
                lambda example: len(example["protein_input_ids"])
                <= self.protein_max_seq_len
                and len(example["drug_input_ids"]) <= self.drug_max_seq_len
            )
            num_filtered_dataset = len(dataset)
            print(
                f"Number of filtered pairs: "
                f"{num_filtered_dataset}/{num_original_dataset} "
                f"({float(num_filtered_dataset)/num_original_dataset*100:.2f}%)"
            )
            dataset_splits[split] = dataset

        data_collator = DataCollatorWithPadding(
            protein_tokenizer=self.protein_tokenizer,
            drug_tokenizer=self.drug_tokenizer,
            padding="max_length",
            protein_max_length=self.protein_max_seq_len,
            drug_max_length=self.drug_max_seq_len,
            return_tensors="pt",
        )

        if "train" in dataset_splits:
            print(f"Setup Train DataLoader")
            self.train_dataloader = DataLoader(
                dataset_splits["train"],
                shuffle=True,
                collate_fn=data_collator,
                batch_size=self.configs.training_configs.batch_size,
                pin_memory=True,
            )
        if "valid" in dataset_splits:
            print(f"Setup Valid DataLoader")
            self.valid_dataloader = DataLoader(
                dataset_splits["valid"],
                shuffle=False,
                collate_fn=data_collator,
                batch_size=self.configs.training_configs.batch_size,
                pin_memory=True,
            )
        print(f"Setup Test DataLoader")
        self.test_dataloader = DataLoader(
            dataset_splits["test"],
            shuffle=False,
            collate_fn=data_collator,
            batch_size=self.configs.training_configs.batch_size,
            pin_memory=True,
        )

    def _setup_run_name(self):
        protein_peft_hyperparameters = (
            self.configs.model_configs.protein_peft_hyperparameters
        )
        drug_peft_hyperparameters = self.configs.model_configs.drug_peft_hyperparameters

        # Group name depends on the dataset and split
        self.group_name = f"{self.configs.dataset_configs.dataset_name}_{self.configs.dataset_configs.split_method}"

        # Run name depends on the hyperparameters
        ## Get hyperparameters
        protein_model_fine_tuning_type = (
            self.configs.model_configs.protein_fine_tuning_type
        )
        drug_model_fine_tuning_type = self.configs.model_configs.drug_fine_tuning_type

        hyperparams = []
        hyperparams += [f"protein_{protein_model_fine_tuning_type}"]
        if protein_peft_hyperparameters:
            for key, value in protein_peft_hyperparameters.items():
                if key not in ["target_modules", "feedforward_modules"]:
                    hyperparams += [f"{key}_{value}"]
        hyperparams += [f"drug_{drug_model_fine_tuning_type}"]
        if drug_peft_hyperparameters:
            for key, value in drug_peft_hyperparameters.items():
                if key not in ["target_modules", "feedforward_modules"]:
                    hyperparams += [f"{key}_{value}"]
        self.run_name = "_".join(hyperparams)

    def setup_training(self):
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            log_with="wandb",
        )
        self.wandb_tracker = None
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name=self.wandb_project,
                init_kwargs={
                    "wandb": {
                        "entity": self.wandb_entity,
                        "name": self.run_name,
                        "group": self.group_name,
                    }
                },
                config=self.configs.dict(),
            )
            self.wandb_tracker: WandBTracker = self.accelerator.get_tracker("wandb")
        self.accelerator.wait_for_everyone()

        if self.train_dataloader is not None:
            # optimizer
            self.optimizer = AdamW(
                params=[
                    param
                    for name, param in self.model.named_parameters()
                    if param.requires_grad
                    and "noise_sigma" not in name  # Handle Balanced MSE loss
                ],
                lr=self.configs.model_configs.model_hyperparameters.learning_rate,
            )

            if self.configs.model_configs.loss_function in ["cosine_balanced_mse"]:
                self.optimizer.add_param_group(
                    {
                        "params": self.model.loss_fn.noise_sigma,
                        "lr": self.configs.model_configs.model_hyperparameters.sigma_lr,
                        "name": "noise_sigma",
                    }
                )

            # lr scheduler
            num_training_steps = (
                len(self.train_dataloader) * self.configs.training_configs.epochs
            )
            warmup_steps_ratio = (
                self.configs.model_configs.model_hyperparameters.warmup_steps_ratio
            )
            self.lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=warmup_steps_ratio,
                num_training_steps=num_training_steps,
            )

            (
                self.model,
                self.train_dataloader,
                self.valid_dataloader,
                self.test_dataloader,
                self.optimizer,
                self.lr_scheduler,
            ) = self.accelerator.prepare(
                self.model,
                self.train_dataloader,
                self.valid_dataloader,
                self.test_dataloader,
                self.optimizer,
                self.lr_scheduler,
            )
        else:
            (
                self.model,
                self.test_dataloader,
            ) = self.accelerator.prepare(
                self.model,
                self.test_dataloader,
            )

        if self.configs.model_configs.checkpoint_path:
            load_trained_model(self.model, self.configs.model_configs, is_training=self.train_dataloader is not None)

    def compute_metrics(self, labels, predictions):
        if self.configs.model_configs.loss_function in [
            "cosine_mse",
            "cosine_balanced_mse",
        ]:
            pkd_range = self.pkd_upper_bound - self.pkd_lower_bound
            labels = (labels + 1) / 2 * pkd_range + self.pkd_lower_bound
            predictions = (predictions + 1) / 2 * pkd_range + self.pkd_lower_bound

        rmse = get_rmse(labels, predictions)
        pearson = get_pearson(labels, predictions)
        spearman = get_spearman(labels, predictions)
        ci = get_ci(labels, predictions)

        return {
            "rmse": rmse,
            "pearson": pearson,
            "spearman": spearman,
            "ci": ci,
        }

    def train(self):
        if self.train_dataloader is None:
            epoch = 0
            best_checkpoint_dir = None
        else:
            best_loss = 999999999
            patience = self.configs.training_configs.patience
            eval_train_every_n_epochs = self.configs.training_configs.epochs // 4
            epochs_no_improve = 0  # Initialize early stopping counter
            best_checkpoint_dir = ""

            print("Trainable params")
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(name)

            for epoch in range(self.configs.training_configs.epochs):
                self.model.train()

                num_train_steps = len(self.train_dataloader)
                progress_bar = tqdm(
                    total=int(num_train_steps // self.gradient_accumulation_steps),
                    position=0,
                    leave=True,
                    disable=not self.accelerator.is_local_main_process,
                )
                total_train_loss = 0
                for train_step, batch in enumerate(self.train_dataloader):
                    with self.accelerator.accumulate(self.model):
                        outputs = self.model(batch)
                        loss = outputs["loss"]

                        # Backpropagate
                        self.accelerator.backward(loss)
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.model.zero_grad()
                        self.optimizer.zero_grad()

                        progress_bar.set_description(f"Epoch {epoch}; Loss: {loss:.4f}")
                        total_train_loss += loss.detach().float()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if self.accelerator.sync_gradients:
                        progress_bar.update(1)

                if (epoch + 1) % eval_train_every_n_epochs == 0:
                    train_metrics = self.test("train")
                else:
                    train_metrics = {
                        "train/loss": total_train_loss / len(self.train_dataloader)
                    }
                # At the end of an epoch, compute metrics
                valid_metrics = self.test("valid")

                if valid_metrics:
                    current_loss = valid_metrics["valid/loss"]
                else:
                    # Just train until the last epoch
                    current_loss = best_loss
                if current_loss <= best_loss:
                    best_loss = current_loss
                    epochs_no_improve = 0
                    # Save the model
                    best_checkpoint_dir = f"step_{epoch}"
                    self.accelerator.save_state(
                        os.path.join(
                            self.outputs_dir, "checkpoint", best_checkpoint_dir
                        )
                    )
                else:
                    epochs_no_improve += 1

                self.accelerator.log(train_metrics | valid_metrics, step=epoch)

                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

            if best_checkpoint_dir:
                self.accelerator.load_state(
                    os.path.join(self.outputs_dir, "checkpoint", best_checkpoint_dir)
                )
                self.accelerator.wait_for_everyone()

        test_metrics = self.test("test", save_prediction=True)
        self.accelerator.log(test_metrics, step=epoch)

        # FIXME: We may want to save the prediction of train and valid for all datasets
        if self.configs.dataset_configs.dataset_name == "BindingDB_filtered":
            # Monkey patch: Just for BindingDB cleaned, we want to check the prediction of train and validation
            train_metrics = self.test("train", save_prediction=True)
            self.accelerator.log(train_metrics, step=epoch)
            valid_metrics = self.test("valid", save_prediction=True)
            self.accelerator.log(valid_metrics, step=epoch)

        if best_checkpoint_dir:
            print("Create a WandB artifact from embedding")
            artifact = wandb.Artifact(best_checkpoint_dir, type="model")
            artifact.add_dir(
                os.path.join(self.outputs_dir, "checkpoint", best_checkpoint_dir)
            )
            wandb.log_artifact(artifact)

    def test(self, split: str, save_prediction=False):
        if split == "train":
            dataloader = self.train_dataloader
        elif split == "valid":
            dataloader = self.valid_dataloader
        elif split == "test":
            dataloader = self.test_dataloader

        if dataloader is None:
            return {}

        total_loss = 0
        all_proteins = []
        all_drugs = []
        all_labels = []
        all_predictions = []

        self.model.eval()

        num_steps = len(dataloader)
        progress_bar = tqdm(
            total=num_steps,
            position=0,
            leave=True,
            disable=not self.accelerator.is_local_main_process,
        )
        for step, batch in enumerate(tqdm(dataloader)):
            with torch.no_grad():
                outputs = self.model(batch)
                loss = outputs["loss"]
                total_loss += loss.detach().float()

                all_proteins += batch["protein_ori_sequences"]
                all_drugs += batch["drug_ori_sequences"]
                if self.configs.model_configs.loss_function in [
                    "cosine_mse",
                    "cosine_balanced_mse",
                ]:
                    all_labels += [batch["labels"]]
                    all_predictions += [outputs["cosine_similarity"]]
                elif self.configs.model_configs.loss_function in ["baseline_mse"]:
                    all_labels += [batch["labels"]]
                    all_predictions += [outputs["logits"]]

            progress_bar.set_description(f"Eval: {split} split")
            progress_bar.update(1)

        all_labels = torch.cat(all_labels, dim=0)
        all_predictions = torch.cat(all_predictions, dim=0)
        performance_metrics = self.compute_metrics(all_labels, all_predictions)
        metrics = {
            f"{split}/loss": total_loss / len(dataloader),
        }
        for metric_name, metric_value in performance_metrics.items():
            metrics[f"{split}/{metric_name}"] = metric_value

        if save_prediction:
            df = pd.DataFrame(columns=["protein", "drug", "label", "prediction"])
            df["protein"] = all_proteins
            df["drug"] = all_drugs

            if self.configs.model_configs.loss_function in [
                "cosine_mse",
                "cosine_balanced_mse",
            ]:
                pkd_range = self.pkd_upper_bound - self.pkd_lower_bound
                all_labels = (all_labels + 1) / 2 * pkd_range + self.pkd_lower_bound
                all_predictions = (
                    all_predictions + 1
                ) / 2 * pkd_range + self.pkd_lower_bound

            df["label"] = all_labels.cpu().numpy().tolist()
            df["prediction"] = all_predictions.cpu().numpy().tolist()
            df.to_csv(os.path.join(self.outputs_dir, f"{split}_prediction.csv"))

            artifact = wandb.Artifact(f"{split}_prediction", type="prediction")
            artifact.add_file(os.path.join(self.outputs_dir, f"{split}_prediction.csv"))
            wandb.log_artifact(artifact)

        return metrics
