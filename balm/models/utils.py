
from typing import Tuple

import os

import torch
from huggingface_hub import hf_hub_download

from balm.configs import ModelConfigs
from balm.models.base_model import BaseModel

def load_trained_model(model: BaseModel, model_configs: ModelConfigs, is_training: bool) -> BaseModel:
    """
    Load a pre-trained model checkpoint and apply necessary adjustments, such as merging adapters 
    if fine-tuning is enabled. Also configures the model for training or evaluation.

    Args:
        model (BaseModel): The model instance to load the checkpoint into.
        model_configs (ModelConfigs): Configuration object containing model-related settings.
        is_training (bool): Flag indicating whether the model is being loaded for training or evaluation.

    Returns:
        BaseModel: The model loaded with the checkpoint and prepared for either training or evaluation.
    """
    # Notify the user about the checkpoint loading
    print(f"Loading checkpoint from {model_configs.checkpoint_path}")

    # Download the checkpoint from the Hugging Face hub using the repository ID and filename
    checkpoint_path = hf_hub_download(
        repo_id=model_configs.checkpoint_path,
        filename="pytorch_model.bin",
        token=os.getenv("HF_TOKEN"),
    )

    # Load the checkpoint into memory and map it to the device of the model
    checkpoint = torch.load(
        checkpoint_path, map_location=model.protein_model.device
    )

    # Retrieve the current state dictionary of the model
    model_state_dict = model.state_dict()

    # Map old parameter names to new ones if necessary, specifically for attention layers
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
            # Update the parameter names to reflect the new layer naming convention
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

    # Apply the name mappings to the checkpoint dictionary
    for old, new in name_mapping.items():
        checkpoint[new] = checkpoint[old]

    # Filter out unnecessary parameters from the checkpoint
    filtered_checkpoint = {
        k: v for k, v in checkpoint.items() if k in model_state_dict
    }
    # Identify any missing keys in the model state dictionary
    missing_keys = [k for k in model_state_dict if k not in checkpoint]

    # Update the model's state dictionary with the filtered checkpoint
    model_state_dict.update(filtered_checkpoint)
    model.load_state_dict(model_state_dict)

    # Notify the user if there are any missing keys in the checkpoint
    if missing_keys:
        print(f"Missing keys in checkpoint: {missing_keys}")

    # Merge fine-tuned adapters with the base model if specified in the configuration
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

    # Configure the model for training or evaluation
    if is_training:
        # Freeze parameters that should not be updated during training
        for name, params in model.named_parameters():
            if "projection" not in name:
                params.requires_grad = False
        model.print_trainable_params()
    else:
        # Freeze all parameters and set the model to evaluation mode
        for name, params in model.named_parameters():
            params.requires_grad = False
        model.eval()

    return model

def load_pretrained_pkd_bounds(checkpoint_path: str) -> Tuple[float, float]:
    """
    Load pre-defined pKd bounds based on the specific checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint.

    Returns:
        Tuple[float, float]: The lower and upper bounds of pKd values for the given dataset.

    Raises:
        ValueError: If the checkpoint path does not match any known pKd scale.
    """
    if "bdb" in checkpoint_path:
        # BindingDB pKd scale
        pkd_lower_bound = 1.999999995657055
        pkd_upper_bound = 10.0
    elif "leakypdb" in checkpoint_path:
        pkd_lower_bound = 0.4
        pkd_upper_bound = 15.22
    elif "mpro" in checkpoint_path:
        pkd_lower_bound = 4.01
        pkd_upper_bound = 10.769216066691143
    else:
        # Raise an error if an unknown pKd scale is encountered
        raise ValueError(
            f"Unknown pKd scale, for {checkpoint_path}"
        )
    
    return pkd_lower_bound, pkd_upper_bound
