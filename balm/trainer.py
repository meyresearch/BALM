import os

import torch
from accelerate import Accelerator
from accelerate.tracking import WandBTracker
from datasets import Dataset
from tdc.multi_pred import DTI
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    DefaultDataCollator,
    get_linear_schedule_with_warmup,
)

import wandb

from .configs import Configs
from .dataset import PairedBindingDataset, create_scaffold_split_dti
from .metrics import get_ci, get_pearson, get_rmse, get_spearman
from .model import BaselineModel, BindingAffinityModel
from .tokenization import pre_tokenize_unique_entities, tokenize_with_lookup


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
            self.model = BindingAffinityModel(self.configs.model_configs)

        (
            self.train_dataloader,
            self.valid_dataloader,
            self.test_dataloader,
        ) = self._load_dataset()

        self._setup_run_name()
        self._setup_training()

    def _load_tokenizers(self):
        protein_tokenizer = AutoTokenizer.from_pretrained(
            self.configs.model_configs.protein_model_name_or_path
        )
        drug_tokenizer = AutoTokenizer.from_pretrained(
            self.configs.model_configs.drug_model_name_or_path
        )

        return protein_tokenizer, drug_tokenizer

    def _get_dataset_split(self, dataset: DTI):
        if self.configs.dataset_configs.split_method == "scaffold":
            # if using scaffold split, we need to use a custom function
            return create_scaffold_split_dti(
                dataset.get_data(),
                seed=self.configs.training_configs.random_seed,
                frac=[0.7, 0.2, 0.1],
                drug_column="Drug",
            )
        else:
            return dataset.get_split(method=self.configs.dataset_configs.split_method)

    def _load_dataset(self) -> dict:
        dataset = DTI(name=self.configs.dataset_configs.dataset_name)
        if self.configs.dataset_configs.harmonize_affinities_mode:
            dataset.harmonize_affinities(
                mode=self.configs.dataset_configs.harmonize_affinities_mode
            )
            # Convert $K_d$ to $pKd$
            dataset.convert_to_log(form="binding")

        # If the loss function is infonce, we need to do a binning
        # based on the selected threshold
        print(
            f"Training with {self.configs.model_configs.loss_function} loss function."
        )

        if self.configs.model_configs.loss_function in ["infonce"]:
            pkd_threshold = self.configs.model_configs.loss_hyperparameters[
                "pkd_threshold"
            ]
            dataset.y = [1 if pkd >= pkd_threshold else 0 for pkd in dataset.y]
        elif self.configs.model_configs.loss_function in ["cosine_mse"]:
            self.pkd_lower_bound = 0
            self.pkd_upper_bound = max(dataset.y)
            print(
                f"Scaling labels: from {self.pkd_lower_bound} - {self.pkd_upper_bound} to -1 to 1"
            )
            dataset.y = [
                (pkd - self.pkd_lower_bound)
                / (self.pkd_upper_bound - self.pkd_lower_bound)
                * 2
                - 1
                for pkd in dataset.y
            ]
        elif self.configs.model_configs.loss_function in ["baseline_mse"]:
            print("Using original pKd")

        dataset_splits = {}
        for split, data_df in self._get_dataset_split(dataset).items():
            # Pre-tokenize unique ligands and proteins for this split
            print(f"Pre-tokenize unique ligands and proteins for {split}")
            protein_tokenized_dict, drug_tokenized_dict = pre_tokenize_unique_entities(
                data_df,
                self.protein_tokenizer,
                self.drug_tokenizer,
                self.protein_max_seq_len,
                self.drug_max_seq_len,
            )

            if self.configs.model_configs.loss_function in ["infonce"]:
                self.num_negative_samples = (
                    self.configs.model_configs.loss_hyperparameters[
                        "num_negative_samples"
                    ]
                )
                self.query_entity_column = (
                    self.configs.model_configs.loss_hyperparameters[
                        "query_entity_column"
                    ]
                )

                dataset = PairedBindingDataset(
                    data_df,
                    protein_tokenized_dict,
                    drug_tokenized_dict,
                    self.num_negative_samples,
                    self.query_entity_column,
                )
            else:
                # Use the optimized tokenization for this split
                dataset = Dataset.from_pandas(data_df).map(
                    lambda x: tokenize_with_lookup(
                        x, protein_tokenized_dict, drug_tokenized_dict
                    ),
                )

            print("Filtering dataset by length")
            print(f"Protein max length: {self.protein_max_seq_len}")
            print(f"Drug max length: {self.drug_max_seq_len}")
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

        data_collator = DefaultDataCollator(return_tensors="pt")

        print(f"Setup Train DataLoader")
        train_dataloader = DataLoader(
            dataset_splits["train"],
            shuffle=True,
            collate_fn=data_collator,
            batch_size=self.configs.training_configs.batch_size,
            pin_memory=True,
        )
        print(f"Setup Valid DataLoader")
        valid_dataloader = DataLoader(
            dataset_splits["valid"],
            shuffle=True,
            collate_fn=data_collator,
            batch_size=self.configs.training_configs.batch_size,
            pin_memory=True,
        )
        print(f"Setup Test DataLoader")
        test_dataloader = DataLoader(
            dataset_splits["test"],
            shuffle=True,
            collate_fn=data_collator,
            batch_size=self.configs.training_configs.batch_size,
            pin_memory=True,
        )

        return train_dataloader, valid_dataloader, test_dataloader

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
        hyperparams += [
            f"lr_{self.configs.model_configs.model_hyperparameters.learning_rate}",
            f"dropout_{self.configs.model_configs.model_hyperparameters.projected_dropout}",
            f"dim_{self.configs.model_configs.model_hyperparameters.projected_size}",
        ]
        self.run_name = "_".join(hyperparams)

    def _setup_training(self):
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
            )
            self.wandb_tracker: WandBTracker = self.accelerator.get_tracker("wandb")
        self.accelerator.wait_for_everyone()

        # optimizer
        self.optimizer = AdamW(
            params=[param for param in self.model.parameters() if param.requires_grad],
            lr=self.configs.model_configs.model_hyperparameters.learning_rate,
        )

        # lr scheduler
        num_training_steps = (
            len(self.train_dataloader) * self.configs.training_configs.epochs
        )
        # warmup_steps_ratio = (
        #     self.configs.model_configs.model_hyperparameters.warmup_steps_ratio
        # )
        # self.lr_scheduler = get_linear_schedule_with_warmup(
        #     optimizer=self.optimizer,
        #     num_warmup_steps=warmup_steps_ratio,
        #     num_training_steps=num_training_steps,
        # )
        self.lr_scheduler = OneCycleLR(
            optimizer=self.optimizer,
            max_lr=self.configs.model_configs.model_hyperparameters.learning_rate * 100,
            total_steps=num_training_steps,
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

    def compute_metrics(self, labels, predictions):
        if self.configs.model_configs.loss_function in ["cosine_mse"]:
            pkd_range = self.pkd_upper_bound - self.pkd_lower_bound
            labels = (labels + 1) / 2 * pkd_range
            predictions = (predictions + 1) / 2 * pkd_range

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

    # def train(self):
    #     for epoch in range(self.configs.training_configs.epochs):
    #         self.model.train()

    #         num_train_steps = len(self.train_dataloader)
    #         progress_bar = tqdm(
    #             total=num_train_steps,
    #             position=0,
    #             leave=True,
    #             disable=not self.accelerator.is_local_main_process,
    #         )
    #         prev_valid_loss = 100000
    #         for train_step, batch in enumerate(self.train_dataloader):
    #             with self.accelerator.accumulate(self.model):
    #                 kwargs = {}
    #                 if self.configs.model_configs.loss_function in ["infonce"]:
    #                     kwargs["query_entity"] = self.query_entity_column
    #                 if self.configs.model_configs.loss_function in [
    #                     "cosine_mse",
    #                     "baseline_mse",
    #                 ]:
    #                     kwargs["labels"] = batch["Y"]

    #                 outputs = self.model(batch, **kwargs)
    #                 loss = outputs["loss"]

    #                 # Backpropagate
    #                 self.accelerator.backward(loss)
    #                 self.optimizer.step()
    #                 self.lr_scheduler.step()
    #                 self.model.zero_grad()
    #                 self.optimizer.zero_grad()

    #                 progress_bar.set_description(f"Epoch {epoch}; Loss: {loss:.4f}")

    #             # Checks if the accelerator has performed an optimization step behind the scenes
    #             if self.accelerator.sync_gradients:
    #                 progress_bar.update(1)

    #         # At the end of an epoch, compute metrics
    #         train_metrics = self.test("train")
    #         valid_metrics = self.test("valid")
    #         self.accelerator.log(train_metrics | valid_metrics, step=epoch)
    #         # Save a checkpoint
    #         if valid_metrics["valid/loss"] < prev_valid_loss:
    #             prev_valid_loss = valid_metrics["valid/loss"]
    #             best_checkpoint_dir = f"step_{epoch}"
    #             self.accelerator.save_state(
    #                 os.path.join(self.outputs_dir, "checkpoint", best_checkpoint_dir)
    #             )

    #     # Load best checkpoint for test evaluation
    #     self.accelerator.load_state(
    #         os.path.join(self.outputs_dir, "checkpoint", best_checkpoint_dir)
    #     )
    #     self.accelerator.wait_for_everyone()

    #     test_metrics = self.test("test")
    #     self.accelerator.log(test_metrics, step=epoch)

    #     print("Create a WandB artifact from embedding")
    #     artifact = wandb.Artifact(self.run_name, type="model")
    #     artifact.add_dir(
    #         os.path.join(self.outputs_dir, "checkpoint", best_checkpoint_dir)
    #     )
    #     wandb.log_artifact(artifact)

    def train(self):
        best_pearson = float("-inf")
        patience = 50
        min_delta = 0.005  # Define minimum change to consider as improvement
        epochs_no_improve = 0  # Initialize early stopping counter
        best_checkpoint_dir = ""

        for epoch in range(self.configs.training_configs.epochs):
            self.model.train()

            num_train_steps = len(self.train_dataloader)
            progress_bar = tqdm(
                total=num_train_steps,
                position=0,
                leave=True,
                disable=not self.accelerator.is_local_main_process,
            )
            prev_valid_loss = 100000
            for train_step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    kwargs = {}
                    if self.configs.model_configs.loss_function in ["infonce"]:
                        kwargs["query_entity"] = self.query_entity_column
                    if self.configs.model_configs.loss_function in [
                        "cosine_mse",
                        "baseline_mse",
                    ]:
                        kwargs["labels"] = batch["Y"]

                    outputs = self.model(batch, **kwargs)
                    loss = outputs["loss"]

                    # Backpropagate
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.model.zero_grad()
                    self.optimizer.zero_grad()

                    progress_bar.set_description(f"Epoch {epoch}; Loss: {loss:.4f}")

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)

            # At the end of an epoch, compute metrics
            train_metrics = self.test("train")
            valid_metrics = self.test("valid")

            current_pearson = valid_metrics["valid/pearson"]
            if current_pearson > best_pearson + min_delta:
                best_pearson = current_pearson
                epochs_no_improve = 0
                # Save the model
                best_checkpoint_dir = f"step_{epoch}"
                self.accelerator.save_state(
                    os.path.join(self.outputs_dir, "checkpoint", best_checkpoint_dir)
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

        test_metrics = self.test("test")
        self.accelerator.log(test_metrics, step=epoch)

        # self.accelerator.log(train_metrics | valid_metrics, step=epoch)

        # Save a checkpoint
        # if valid_metrics["valid/loss"] < prev_valid_loss:
        #     prev_valid_loss = valid_metrics["valid/loss"]
        #     best_checkpoint_dir = f"step_{epoch}"
        #     self.accelerator.save_state(
        #         os.path.join(self.outputs_dir, "checkpoint", best_checkpoint_dir)
        #     )

        # Load best checkpoint for test evaluation
        # self.accelerator.load_state(
        #     os.path.join(self.outputs_dir, "checkpoint", best_checkpoint_dir)
        # )
        # self.accelerator.wait_for_everyone()

        # test_metrics = self.test("test")
        # self.accelerator.log(test_metrics, step=epoch)

        print("Create a WandB artifact from embedding")
        artifact = wandb.Artifact(self.run_name, type="model")
        artifact.add_dir(
            os.path.join(self.outputs_dir, "checkpoint", best_checkpoint_dir)
        )
        wandb.log_artifact(artifact)

    def test(self, split: str):
        if split == "train":
            dataloader = self.train_dataloader
        elif split == "valid":
            dataloader = self.valid_dataloader
        elif split == "test":
            dataloader = self.test_dataloader

        total_loss = 0
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
            kwargs = {}
            if self.configs.model_configs.loss_function in ["infonce"]:
                kwargs["query_entity"] = self.query_entity_column
            if self.configs.model_configs.loss_function in [
                "cosine_mse",
                "baseline_mse",
            ]:
                kwargs["labels"] = batch["Y"]

            with torch.no_grad():
                outputs = self.model(batch, **kwargs)
                loss = outputs["loss"]
                total_loss += loss.detach().float()

                if self.configs.model_configs.loss_function in ["cosine_mse"]:
                    all_labels += [kwargs["labels"]]
                    all_predictions += [outputs["cosine_similarity"]]
                elif self.configs.model_configs.loss_function in ["baseline_mse"]:
                    all_labels += [kwargs["labels"]]
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

        return metrics
