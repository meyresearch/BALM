from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from tdc.multi_pred import DTI


from balm.datasets.bindingdb_filtered import create_scaffold_split_dti
from balm.configs import DatasetConfigs, TrainingConfigs


class DataCollatorWithPadding:
    def __init__(
        self,
        protein_tokenizer: PreTrainedTokenizerBase,
        drug_tokenizer: PreTrainedTokenizerBase,
        padding: Union[bool, str, PaddingStrategy] = True,
        protein_max_length: Optional[int] = None,
        drug_max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: str = "pt",
    ):
        self.protein_tokenizer = protein_tokenizer
        self.drug_tokenizer = drug_tokenizer
        self.padding = padding
        self.protein_max_length = protein_max_length
        self.drug_max_length = drug_max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Extract 'protein_input_ids' and prepare them for padding
        protein_features = [
            {"input_ids": feature["protein_input_ids"]} for feature in features
        ]

        # Pad 'protein_input_ids' and ensure they're named correctly after padding
        padded_protein_features = self.protein_tokenizer.pad(
            protein_features,
            padding=self.padding,
            max_length=self.protein_max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        # Extract 'drug_input_ids' and prepare them for padding
        drug_features = [
            {"input_ids": feature["drug_input_ids"]} for feature in features
        ]

        # Pad 'drug_input_ids' and ensure they're named correctly after padding
        padded_drug_features = self.drug_tokenizer.pad(
            drug_features,
            padding=self.padding,
            max_length=self.drug_max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        batch = {
            "protein_ori_sequences": [
                feature["protein_ori_sequences"] for feature in features
            ],
            "drug_ori_sequences": [
                feature["drug_ori_sequences"] for feature in features
            ],
            "protein_input_ids": padded_protein_features["input_ids"],
            "protein_attention_mask": padded_protein_features["attention_mask"],
            "drug_input_ids": padded_drug_features["input_ids"],
            "drug_attention_mask": padded_drug_features["attention_mask"],
            "labels": torch.stack([torch.tensor(feature["Y"]) for feature in features]),
        }

        return batch


def get_dataset_split(dataset_configs: DatasetConfigs, training_configs: TrainingConfigs, dataset: Union[DTI]):
    if (
        dataset_configs.split_method == "scaffold"
        and dataset_configs.dataset_name in ["DTI_BindingDB_Kd"]
    ):
        # if using scaffold split, we need to use a custom function
        return create_scaffold_split_dti(
            method=dataset_configs.split_method,
            seed=training_configs.random_seed,
            frac=[0.7, 0.2, 0.1],
            drug_column="Drug",
        )

    else:
        return dataset.get_split(
            method=dataset_configs.split_method,
            seed=training_configs.random_seed,
        )
