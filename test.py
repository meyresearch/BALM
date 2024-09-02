import argparse
import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv("env/.env")

import time
import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from balm import common_utils
from balm.configs import Configs
from balm.dataset import DataCollatorWithPadding
from balm.metrics import get_ci, get_pearson, get_rmse, get_spearman
from balm.model import BindingAffinityModel
from balm.tokenization import pre_tokenize_unique_entities, tokenize_with_lookup


def argument_parser():
    """
    Parses command-line arguments for the BALM testing script.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="BALM")
    parser.add_argument("--config_filepath", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--test_data", type=str, required=True, help="Path to the test data CSV file.")
    parser.add_argument("--pkd_upper_bound", type=float, default=10.0, help="Upper bound for pKd scaling.")
    parser.add_argument("--pkd_lower_bound", type=float, default=1.999999995657055, help="Lower bound for pKd scaling.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint/", help="Directory containing model checkpoints.")
    args = parser.parse_args()
    return args


def get_checkpoint_name(configs: Configs):
    """
    Constructs the checkpoint name based on the configuration hyperparameters.

    Args:
        configs (Configs): Configuration object containing model settings.

    Returns:
        str: The constructed checkpoint name.
    """
    protein_peft_hyperparameters = configs.model_configs.protein_peft_hyperparameters
    drug_peft_hyperparameters = configs.model_configs.drug_peft_hyperparameters

    # Build the run name based on the fine-tuning types and hyperparameters
    hyperparams = []
    hyperparams += [f"protein_{configs.model_configs.protein_fine_tuning_type}"]
    if protein_peft_hyperparameters:
        for key, value in protein_peft_hyperparameters.items():
            if key not in ["target_modules", "feedforward_modules"]:
                hyperparams += [f"{key}_{value}"]
    hyperparams += [f"drug_{configs.model_configs.drug_fine_tuning_type}"]
    if drug_peft_hyperparameters:
        for key, value in drug_peft_hyperparameters.items():
            if key not in ["target_modules", "feedforward_modules"]:
                hyperparams += [f"{key}_{value}"]
    hyperparams += [
        f"lr_{configs.model_configs.model_hyperparameters.learning_rate}",
        f"dropout_{configs.model_configs.model_hyperparameters.projected_dropout}",
        f"dim_{configs.model_configs.model_hyperparameters.projected_size}",
    ]
    run_name = "_".join(hyperparams)
    return run_name


def load_model(configs, checkpoint_dir):
    """
    Loads the model from the specified checkpoint directory.

    Args:
        configs (Configs): Configuration object containing model settings.
        checkpoint_dir (str): Directory where model checkpoints are stored.

    Returns:
        BindingAffinityModel: The loaded and prepared model.
    """
    model = BindingAffinityModel(configs.model_configs)
    model = model.to("mps")
    checkpoint_name = get_checkpoint_name(configs)
    print(f"Loading checkpoint from {os.path.join(checkpoint_dir, checkpoint_name)}")
    checkpoint = torch.load(
        os.path.join(checkpoint_dir, checkpoint_name, "pytorch_model.bin"),
        map_location=torch.device("mps"),
    )

    model.load_state_dict(checkpoint)
    model = model.eval()

    # Merge PEFT and base model
    if configs.model_configs.protein_fine_tuning_type in ["lora", "lokr", "loha", "ia3"]:
        model.protein_model.merge_and_unload()
    if configs.model_configs.drug_fine_tuning_type in ["lora", "lokr", "loha", "ia3"]:
        model.drug_model.merge_and_unload()

    return model


def load_tokenizers(configs):
    """
    Loads the tokenizers for protein and drug sequences.

    Args:
        configs (Configs): Configuration object containing model settings.

    Returns:
        tuple: A tuple containing the protein and drug tokenizers.
    """
    protein_tokenizer = AutoTokenizer.from_pretrained(configs.model_configs.protein_model_name_or_path)
    drug_tokenizer = AutoTokenizer.from_pretrained(configs.model_configs.drug_model_name_or_path)

    return protein_tokenizer, drug_tokenizer


def load_data(
    test_data,
    batch_size,
    protein_tokenizer,
    drug_tokenizer,
    protein_max_seq_len,
    drug_max_seq_len,
):
    """
    Loads and prepares the test dataset for evaluation.

    Args:
        test_data (str): Path to the test data CSV file.
        batch_size (int): Batch size for data loading.
        protein_tokenizer (PreTrainedTokenizer): Tokenizer for protein sequences.
        drug_tokenizer (PreTrainedTokenizer): Tokenizer for drug sequences.
        protein_max_seq_len (int): Maximum sequence length for protein tokens.
        drug_max_seq_len (int): Maximum sequence length for drug tokens.

    Returns:
        DataLoader: DataLoader for the prepared test dataset.
    """
    df = pd.read_csv(test_data)
    protein_tokenized_dict, drug_tokenized_dict = pre_tokenize_unique_entities(
        df,
        protein_tokenizer,
        drug_tokenizer,
    )

    dataset = Dataset.from_pandas(df).map(
        lambda x: tokenize_with_lookup(x, protein_tokenized_dict, drug_tokenized_dict),
    )

    data_collator = DataCollatorWithPadding(
        protein_tokenizer=protein_tokenizer,
        drug_tokenizer=drug_tokenizer,
        padding="max_length",
        protein_max_length=protein_max_seq_len,
        drug_max_length=drug_max_seq_len,
        return_tensors="pt",
    )

    print(f"Setup Train DataLoader")
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size,
        pin_memory=True,
    )

    return dataloader


def compute_metrics(labels, predictions, pkd_upper_bound, pkd_lower_bound):
    """
    Computes performance metrics including RMSE, Pearson, Spearman, and CI.

    Args:
        labels (Tensor): Ground truth labels.
        predictions (Tensor): Model predictions.
        pkd_upper_bound (float): Upper bound for pKd scaling.
        pkd_lower_bound (float): Lower bound for pKd scaling.

    Returns:
        dict: A dictionary containing the computed metrics.
    """
    pkd_range = pkd_upper_bound - pkd_lower_bound
    labels = (labels + 1) / 2 * pkd_range + pkd_lower_bound
    predictions = (predictions + 1) / 2 * pkd_range + pkd_lower_bound

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


def main():
    """
    Main function to execute the testing process for BALM.

    It performs the following steps:
    1. Parses command-line arguments.
    2. Loads configuration settings and model checkpoints.
    3. Prepares the test dataset.
    4. Evaluates the model on the test data and computes performance metrics.
    """
    args = argument_parser()
    config_filepath = args.config_filepath
    configs = Configs(**common_utils.load_yaml(config_filepath))

    protein_max_seq_len = configs.model_configs.model_hyperparameters.protein_max_seq_len
    drug_max_seq_len = configs.model_configs.model_hyperparameters.drug_max_seq_len
    protein_tokenizer, drug_tokenizer = load_tokenizers(configs)

    model = load_model(configs, args.checkpoint_dir)
    dataloader = load_data(
        args.test_data,
        configs.training_configs.batch_size,
        protein_tokenizer,
        drug_tokenizer,
        protein_max_seq_len,
        drug_max_seq_len,
    )

    pkd_upper_bound = args.pkd_upper_bound
    pkd_lower_bound = args.pkd_lower_bound

    start = time.time()
    all_labels = []
    all_predictions = []
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            batch = {key: value.to(model.protein_model.device) for key, value in batch.items()}
            outputs = model(batch)
            all_labels += [batch["labels"]]
            all_predictions += [outputs["cosine_similarity"]]
            if step % 10:
                print(
                    f"Time elapsed: {time.time()-start}s ; Processed: {step * configs.training_configs.batch_size}"
                )
    end = time.time()
    print(f"Finished! Time taken: {end - start}s")

    all_labels = torch.cat(all_labels, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)
    performance_metrics = compute_metrics(
        all_labels, all_predictions, pkd_upper_bound, pkd_lower_bound
    )
    print(performance_metrics)


if __name__ == "__main__":
    main()
