import pandas as pd
from transformers import PreTrainedTokenizer


def pre_tokenize_unique_entities(
    dataset: pd.DataFrame,
    protein_tokenizer: PreTrainedTokenizer,
    drug_tokenizer: PreTrainedTokenizer,
):
    unique_proteins = dataset["Target"].unique().tolist()
    unique_drugs = dataset["Drug"].unique().tolist()

    tokenized_proteins = protein_tokenizer(unique_proteins)
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

    # Check if there's non string values in unique_drugs
    # Count how many unique drugs are not strings
    non_string_drugs = sum([not isinstance(drug, str) for drug in unique_drugs])
    if non_string_drugs > 0:
        # Print the non string drugs
        print("Non string drugs:")
        print([drug for drug in unique_drugs if not isinstance(drug, str)])

    tokenized_drugs = drug_tokenizer(unique_drugs)
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
        "protein_ori_sequences": examples["Target"],
        "drug_ori_sequences": examples["Drug"],
        "protein_input_ids": protein_input["input_ids"],
        "drug_input_ids": drug_input["input_ids"],
        "protein_attention_mask": protein_input["attention_mask"],
        "drug_attention_mask": drug_input["attention_mask"],
    }
