from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class FineTuningType(str, Enum):
    """
    baseline: Only for baseline model (Concatenated Embedding + linear projection)
    projection: Common fine tuning technique: only tuning Linear projection
    """

    baseline = "baseline"
    projection = "projection"
    lora = "lora"
    lokr = "lokr"
    loha = "loha"
    ia3 = "ia3"


class ModelHyperparameters(BaseModel):
    learning_rate: float = 0.001
    protein_max_seq_len: int = 1024
    drug_max_seq_len: int = 512
    warmup_steps_ratio: float = 0.06
    gradient_accumulation_steps: int = 32
    projected_size: int = 256
    projected_dropout: float = 0.5
    relu_before_cosine: bool = False
    init_noise_sigma: float = 1
    sigma_lr: float = 0.01


class ModelConfigs(BaseModel):
    protein_model_name_or_path: str
    drug_model_name_or_path: str
    checkpoint_path: Optional[str] = None
    model_hyperparameters: ModelHyperparameters
    protein_fine_tuning_type: Optional[FineTuningType]
    drug_fine_tuning_type: Optional[FineTuningType]
    protein_peft_hyperparameters: Optional[dict]
    drug_peft_hyperparameters: Optional[dict]
    loss_function: str


class DatasetConfigs(BaseModel):
    dataset_name: str
    harmonize_affinities_mode: Optional[str]
    split_method: str = "random"
    train_ratio: Optional[float] = None


class TrainingConfigs(BaseModel):
    random_seed: int = 1234
    device: int = 0
    epochs: int = 1
    batch_size: int = 4
    patience: int = 100
    min_delta: int = 0.005
    outputs_dir: str = "outputs"


class Configs(BaseModel):
    model_configs: ModelConfigs
    dataset_configs: DatasetConfigs
    training_configs: TrainingConfigs
