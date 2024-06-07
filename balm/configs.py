from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class FineTuningType(str, Enum):
    """
    baseline: Only for baseline model (Concatenated Embedding + linear projection)
    projection: Common fine tuning technique: only tuning linear projection
    """

    baseline = "baseline"
    projection = "projection"
    lora = "lora"
    lokr = "lokr"
    loha = "loha"
    ia3 = "ia3"


class ModelHyperparameters(BaseModel):
    learning_rate: float
    protein_max_seq_len: int
    drug_max_seq_len: int
    warmup_steps_ratio: float
    gradient_accumulation_steps: int
    projected_size: int
    projected_dropout: float


class ModelConfigs(BaseModel):
    protein_model_name_or_path: str
    drug_model_name_or_path: str
    model_hyperparameters: ModelHyperparameters
    protein_fine_tuning_type: Optional[FineTuningType]
    drug_fine_tuning_type: Optional[FineTuningType]
    protein_peft_hyperparameters: Optional[dict]
    drug_peft_hyperparameters: Optional[dict]
    loss_function: str
    loss_hyperparameters: Optional[dict]


class DatasetConfigs(BaseModel):
    dataset_name: str
    harmonize_affinities_mode: Optional[str]
    split_method: str = "random"


class TrainingConfigs(BaseModel):
    random_seed: int = 1234
    device: int = 0
    epochs: int = 1
    batch_size: int = 256
    outputs_dir: str = "outputs"


class Configs(BaseModel):
    model_configs: ModelConfigs
    dataset_configs: DatasetConfigs
    training_configs: TrainingConfigs
