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
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
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
    """
    The Trainer class handles the training, validation, and testing processes for the models.
    It supports setting up datasets, initializing models, and managing the training loop with
    early stopping and learning rate scheduling.

    Attributes:
        configs (Configs): Configuration object with all necessary hyperparameters and settings.
        wandb_entity (str): Weights & Biases entity name.
        wandb_project (str): Weights & Biases project name.
        outputs_dir (str): Directory where output files such as checkpoints and logs are saved.
    """

    def __init__(
        self, configs: Configs, wandb_entity: str, wandb_project: str, outputs_dir: str
    ):
        """
        Initialize the Trainer with the provided configurations, Weights & Biases settings, 
        and output directory.

        Args:
            configs (Configs): Configuration object.
            wandb_entity (str): Weights & Biases entity name.
            wandb_project (str): Weights & Biases project name.
            outputs_dir (str): Directory where outputs are saved.
        """
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

        # Load the tokenizers for protein and drug sequences
        self.protein_tokenizer, self.drug_tokenizer = self._load_tokenizers()

        # Determine which model to use based on fine-tuning type
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
        """
        Load the tokenizers for protein and drug sequences based on the model paths.

        Returns:
            Tuple: (protein_tokenizer, drug_tokenizer)
        """
        protein_tokenizer = AutoTokenizer.from_pretrained(
            self.configs.model_configs.protein_model_name_or_path
        )
        drug_tokenizer = AutoTokenizer.from_pretrained(
            self.configs.model_configs.drug_model_name_or_path
        )

        return protein_tokenizer, drug_tokenizer

    def set_pkd_bounds(self, dataset):
        """
        Set the pKd bounds for scaling the labels in the dataset. If a checkpoint is loaded 
        for a zero-shot experiment, the bounds are loaded from the checkpoint.

        Args:
            dataset (Dataset): The dataset containing the pKd labels.
        """
        self.pkd_lower_bound = min(dataset.y)
        self.pkd_upper_bound = max(dataset.y)

        # Load pKd bounds from a trained model if performing a zero-shot experiment
        if self.configs.model_configs.checkpoint_path:
            if self.configs.dataset_configs.train_ratio == 0.0:
                self.pkd_lower_bound, self.pkd_upper_bound = load_pretrained_pkd_bounds(self.configs.model_configs.checkpoint_path)
        
        print(
            f"Scaling labels: from {self.pkd_lower_bound} - {self.pkd_upper_bound} to -1 to 1"
        )

    def set_dataset(self, *args, **kwargs) -> dict:
        """
        Prepare and set up the dataset for training, validation, and testing. This includes
        pre-tokenization, filtering based on sequence length, and setting up DataLoaders.

        Returns:
            dict: Dictionary containing the dataset splits (train, valid, test).
        """
        dataset = factories.get_dataset(self.configs.dataset_configs.dataset_name)

        print(
            f"Training with {self.configs.model_configs.loss_function} loss function."
        )

        # Apply pKd scaling if using cosine MSE loss
        if self.configs.model_configs.loss_function == "cosine_mse":
            self.set_pkd_bounds(dataset)

            if self.pkd_upper_bound == self.pkd_lower_bound:
                # Handle case where all labels are the same
                dataset.y = [0 for _ in dataset.y]
            else:
                dataset.y = [
                    (pkd - self.pkd_lower_bound)
                    / (self.pkd_upper_bound - self.pkd_lower_bound)
                    * 2
                    - 1
                    for pkd in dataset.y
                ]
            # Preprocess Y column for non-TDC datasets using the same pKd scaling
            if not self.configs.dataset_configs.dataset_name.startswith("DTI_"):
                if self.pkd_upper_bound == self.pkd_lower_bound:
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

        # Filter the dataset by sequence length
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

            # Tokenize the dataset and filter by sequence length
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

        # Create data collator to handle padding during batching
        data_collator = DataCollatorWithPadding(
            protein_tokenizer=self.protein_tokenizer,
            drug_tokenizer=self.drug_tokenizer,
            padding="max_length",
            protein_max_length=self.protein_max_seq_len,
            drug_max_length=self.drug_max_seq_len,
            return_tensors="pt",
        )

        # Setup DataLoaders for train, valid, and test splits
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
        """
        Setup the run name and group name for the Weights & Biases tracker based on
        the dataset, split method, and model hyperparameters.
        """
        protein_peft_hyperparameters = (
            self.configs.model_configs.protein_peft_hyperparameters
        )
        drug_peft_hyperparameters = self.configs.model_configs.drug_peft_hyperparameters

        # Group name depends on the dataset and split method
        self.group_name = f"{self.configs.dataset_configs.dataset_name}_{self.configs.dataset_configs.split_method}"

        # Run name depends on the fine-tuning type and other relevant hyperparameters
        hyperparams = []
        hyperparams += [f"protein_{self.configs.model_configs.protein_fine_tuning_type}"]
        if protein_peft_hyperparameters:
            for key, value in protein_peft_hyperparameters.items():
                if key not in ["target_modules", "feedforward_modules"]:
                    hyperparams += [f"{key}_{value}"]
        hyperparams += [f"drug_{self.configs.model_configs.drug_fine_tuning_type}"]
        if drug_peft_hyperparameters:
            for key, value in drug_peft_hyperparameters.items():
                if key not in ["target_modules", "feedforward_modules"]:
                    hyperparams += [f"{key}_{value}"]
        self.run_name = "_".join(hyperparams)

    def setup_training(self):
        """
        Setup the training environment, including initializing the Accelerator, WandB tracker, 
        optimizer, and learning rate scheduler. Prepares the model and dataloaders for training.
        """
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
            # Initialize optimizer with parameters that require gradients
            self.optimizer = AdamW(
                params=[
                    param
                    for name, param in self.model.named_parameters()
                    if param.requires_grad
                    and "noise_sigma" not in name  # Handle Balanced MSE loss
                ],
                lr=self.configs.model_configs.model_hyperparameters.learning_rate,
            )


            # Setup learning rate scheduler
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

            # Prepare model, dataloaders, optimizer, and scheduler for training
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
            # If only testing, prepare the model and test dataloader
            (
                self.model,
                self.test_dataloader,
            ) = self.accelerator.prepare(
                self.model,
                self.test_dataloader,
            )

        # Load a trained model from checkpoint if specified
        if self.configs.model_configs.checkpoint_path:
            load_trained_model(self.model, self.configs.model_configs, is_training=self.train_dataloader is not None)

    def compute_metrics(self, labels, predictions):
        """
        Compute evaluation metrics including RMSE, Pearson, Spearman, and CI.

        Args:
            labels (Tensor): True labels.
            predictions (Tensor): Predicted values.

        Returns:
            dict: Dictionary containing the computed metrics.
        """
        if self.configs.model_configs.loss_function in [
            "cosine_mse"
        ]:
            # Rescale predictions and labels back to the original pKd range
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
        """
        Execute the training loop, handling early stopping, checkpoint saving, and logging metrics.
        """
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

                        # Backpropagation
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
                # At the end of an epoch, compute validation metrics
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

            # Reload the best model checkpoint
            if best_checkpoint_dir:
                self.accelerator.load_state(
                    os.path.join(self.outputs_dir, "checkpoint", best_checkpoint_dir)
                )
                self.accelerator.wait_for_everyone()

        # Compute test metrics and log results
        test_metrics = self.test("test", save_prediction=True)
        self.accelerator.log(test_metrics, step=epoch)

        # For specific datasets, also save predictions for train and validation splits
        if self.configs.dataset_configs.dataset_name == "BindingDB_filtered":
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
        """
        Evaluate the model on the specified dataset split and optionally save predictions.

        Args:
            split (str): The dataset split to evaluate on ('train', 'valid', 'test').
            save_prediction (bool): Whether to save the predictions as a CSV file.

        Returns:
            dict: Dictionary containing the evaluation metrics for the specified split.
        """
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

                # Collect predictions and labels for metric computation
                all_proteins += batch["protein_ori_sequences"]
                all_drugs += batch["drug_ori_sequences"]
                if self.configs.model_configs.loss_function in [
                    "cosine_mse"
                ]:
                    all_labels += [batch["labels"]]
                    all_predictions += [outputs["cosine_similarity"]]
                elif self.configs.model_configs.loss_function in ["baseline_mse"]:
                    all_labels += [batch["labels"]]
                    all_predictions += [outputs["logits"]]

            progress_bar.set_description(f"Eval: {split} split")
            progress_bar.update(1)

        # Concatenate all predictions and labels across batches
        all_labels = torch.cat(all_labels, dim=0)
        all_predictions = torch.cat(all_predictions, dim=0)
        performance_metrics = self.compute_metrics(all_labels, all_predictions)
        metrics = {
            f"{split}/loss": total_loss / len(dataloader),
        }
        for metric_name, metric_value in performance_metrics.items():
            metrics[f"{split}/{metric_name}"] = metric_value

        if save_prediction:
            # Save predictions and labels to a CSV file
            df = pd.DataFrame(columns=["protein", "drug", "label", "prediction"])
            df["protein"] = all_proteins
            df["drug"] = all_drugs

            if self.configs.model_configs.loss_function in [
                "cosine_mse"
            ]:
                pkd_range = self.pkd_upper_bound - self.pkd_lower_bound
                all_labels = (all_labels + 1) / 2 * pkd_range + self.pkd_lower_bound
                all_predictions = (
                    all_predictions + 1
                ) / 2 * pkd_range + self.pkd_lower_bound

            df["label"] = all_labels.cpu().numpy().tolist()
            df["prediction"] = all_predictions.cpu().numpy().tolist()
            df.to_csv(os.path.join(self.outputs_dir, f"{split}_prediction.csv"))

            # Log the predictions as a WandB artifact
            artifact = wandb.Artifact(f"{split}_prediction", type="prediction")
            artifact.add_file(os.path.join(self.outputs_dir, f"{split}_prediction.csv"))
            wandb.log_artifact(artifact)

        return metrics
