
from typing import Tuple

import os

import torch
from huggingface_hub import hf_hub_download

from balm.configs import ModelConfigs
from balm.models.base_model import BaseModel


def load_trained_model(model: BaseModel, model_configs: ModelConfigs, is_training: bool):
    print(f"Loading checkpoint from {model_configs.checkpoint_path}")

    checkpoint_path = hf_hub_download(
        repo_id=model_configs.checkpoint_path,
        filename="pytorch_model.bin",
        token=os.getenv("HF_DOWNLOAD_TOKEN"),
    )

    checkpoint = torch.load(
        checkpoint_path, map_location=model.protein_model.device
    )

    model_state_dict = model.state_dict()

    name_mapping = {}
    for name, param in checkpoint.items():
        if (
            "protein_model.base_model.model.encoder.layer" in name
            or "drug_model.base_model.model.encoder.layer" in name
        ) and (
            "attention.self.query.weight" in name
            or "attention.self.query.bias" in name
            or "attention.self.key.weight" in name
            or "attention.self.key.bias" in name
            or "attention.self.value.weight" in name
            or "attention.self.value.bias" in name
        ):
            new_name = name.replace(
                "attention.self.query.weight",
                "attention.self.query.base_layer.weight",
            )
            new_name = new_name.replace(
                "attention.self.query.bias", "attention.self.query.base_layer.bias"
            )
            new_name = new_name.replace(
                "attention.self.key.weight", "attention.self.key.base_layer.weight"
            )
            new_name = new_name.replace(
                "attention.self.key.bias", "attention.self.key.base_layer.bias"
            )
            new_name = new_name.replace(
                "attention.self.value.weight",
                "attention.self.value.base_layer.weight",
            )
            new_name = new_name.replace(
                "attention.self.value.bias", "attention.self.value.base_layer.bias"
            )
            name_mapping[name] = new_name

    for old, new in name_mapping.items():
        checkpoint[new] = checkpoint[old]

    filtered_checkpoint = {
        k: v for k, v in checkpoint.items() if k in model_state_dict
    }
    missing_keys = [k for k in model_state_dict if k not in checkpoint]

    model_state_dict.update(filtered_checkpoint)
    model.load_state_dict(model_state_dict)

    if missing_keys:
        print(f"Missing keys in checkpoint: {missing_keys}")

    # Merge PEFT and base model if specified
    if model_configs.protein_fine_tuning_type in [
        "lora",
        "lokr",
        "loha",
        "ia3",
    ]:
        print("Merging protein model with its adapter")
        model.protein_model.merge_and_unload()
    if model_configs.drug_fine_tuning_type in [
        "lora",
        "lokr",
        "loha",
        "ia3",
    ]:
        print("Merging drug model with its adapter")
        model.drug_model.merge_and_unload()

    # Evaluate the model if not in training mode
    if is_training:
        for name, params in model.named_parameters():
            if "projection" not in name:
                params.requires_grad = False
        model.print_trainable_params()
    else:
        for name, params in model.named_parameters():
            params.requires_grad = False
        model.eval()

    return model


def load_pretrained_pkd_bounds(checkpoint_path: str) -> Tuple[float, float]:
    if "bdb" in checkpoint_path:
        # BindingDB
        pkd_lower_bound = 1.999999995657055
        pkd_upper_bound = 10.0
    elif "leakypdb" in checkpoint_path:
        pkd_lower_bound = 0.4
        pkd_upper_bound = 15.22
    elif "mpro" in checkpoint_path:
        pkd_lower_bound = 4.01
        pkd_upper_bound = 10.769216066691143
    else:
        raise ValueError(
            f"Unknown pKd scale, for {checkpoint_path}"
        )
    
    return pkd_lower_bound, pkd_upper_bound