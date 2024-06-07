import pandas as pd
from transformers import PreTrainedTokenizer


def tokenize(
    examples,
    protein_tokenizer: PreTrainedTokenizer,
    drug_tokenizer: PreTrainedTokenizer,
    max_seq_len: int,
):
    protein_input = protein_tokenizer(
        examples["Target"],
        truncation=True,
        padding="max_length",
        max_length=max_seq_len,
    )
    drug_input = drug_tokenizer(
        examples["Drug"], truncation=True, padding="max_length", max_length=max_seq_len
    )
    return {
        "protein_input_ids": protein_input["input_ids"],
        "drug_input_ids": drug_input["input_ids"],
        "protein_attention_mask": protein_input["attention_mask"],
        "drug_attention_mask": drug_input["attention_mask"],
    }


def tokenize_one_entity(
    examples,
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int,
):
    tokenized_input = tokenizer(
        examples,
        truncation=True,
        padding="max_length",
        max_length=max_seq_len,
    )
    return {
        "input_ids": tokenized_input["input_ids"],
        "attention_mask": tokenized_input["attention_mask"],
    }


def pre_tokenize_unique_entities(
    dataset: pd.DataFrame,
    protein_tokenizer: PreTrainedTokenizer,
    drug_tokenizer: PreTrainedTokenizer,
    protein_max_seq_len: int,
    drug_max_seq_len: int,
):
    unique_proteins = dataset["Target"].unique().tolist()
    unique_drugs = dataset["Drug"].unique().tolist()

    tokenized_proteins = protein_tokenizer(
        unique_proteins,
        # truncation=True,
        padding="max_length",
        max_length=protein_max_seq_len,
    )
    protein_tokenized_dict = {
        protein: {
            "input_ids": tokenized_protein_input_ids,
            "attention_mask": tokenized_protein_attention_mask,
        }
        for protein, tokenized_protein_input_ids, tokenized_protein_attention_mask in zip(
            unique_proteins,
            tokenized_proteins["input_ids"],
            tokenized_proteins["attention_mask"],
        )
    }

    tokenized_drugs = drug_tokenizer(
        unique_drugs,
        # truncation=True,
        padding="max_length",
        max_length=drug_max_seq_len,
    )
    drug_tokenized_dict = {
        drug: {
            "input_ids": tokenized_drug_input_ids,
            "attention_mask": tokenized_drug_attention_mask,
        }
        for drug, tokenized_drug_input_ids, tokenized_drug_attention_mask in zip(
            unique_drugs,
            tokenized_drugs["input_ids"],
            tokenized_drugs["attention_mask"],
        )
    }

    return protein_tokenized_dict, drug_tokenized_dict


def tokenize_with_lookup(examples, protein_tokenized_dict, drug_tokenized_dict):
    protein_input = protein_tokenized_dict[examples["Target"]]
    drug_input = drug_tokenized_dict[examples["Drug"]]

    return {
        "protein_input_ids": protein_input["input_ids"],
        "drug_input_ids": drug_input["input_ids"],
        "protein_attention_mask": protein_input["attention_mask"],
        "drug_attention_mask": drug_input["attention_mask"],
    }
