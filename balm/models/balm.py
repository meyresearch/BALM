import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

from peft import (
    IA3Config,
    LoHaConfig,
    LoKrConfig,
    LoraConfig,
    TaskType,
    get_peft_model,
)

from balm.configs import ModelConfigs
from balm.models.base_model import BaseModel


PEFT_TYPE_TO_CONFIG_MAPPING = {
    "lora": LoraConfig,
    "loha": LoHaConfig,
    "lokr": LoKrConfig,
    "ia3": IA3Config,
}

def get_peft_config(peft_type):
    return PEFT_TYPE_TO_CONFIG_MAPPING[peft_type]


class BALM(BaseModel):
    
    """
    BALM Model using the protein and ligand laguage models and projecting them to a common space using cosine loss.
    Args:
        model_configs (ModelConfigs): The configuration object for the model.
        protein_embedding_size (int): The size of the protein embedding. Default is 640.
        drug_embedding_size (int): The size of the drug embedding. Default is 384.
    """

    def __init__(
        self,
        model_configs: ModelConfigs,
        protein_embedding_size=640,
        drug_embedding_size=384,
    ):
        super(BALM, self).__init__(
            model_configs, protein_embedding_size, drug_embedding_size
        )

        self.relu_before_cosine = (
            self.model_configs.model_hyperparameters.relu_before_cosine
        )

        if model_configs.protein_peft_hyperparameters:
            self.protein_peft_config = get_peft_config(
                model_configs.protein_fine_tuning_type
            )(
                **model_configs.protein_peft_hyperparameters,
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.protein_model = get_peft_model(
                self.protein_model, self.protein_peft_config
            )
            self.protein_model.print_trainable_parameters()
        if model_configs.drug_peft_hyperparameters:
            self.drug_peft_config = get_peft_config(
                model_configs.drug_fine_tuning_type
            )(
                **model_configs.drug_peft_hyperparameters,
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.drug_model = get_peft_model(self.drug_model, self.drug_peft_config)
            self.drug_model.print_trainable_parameters()

        self.protein_projection = nn.Linear(
            self.protein_embedding_size,
            model_configs.model_hyperparameters.projected_size,
        )
        self.drug_projection = nn.Linear(
            self.drug_embedding_size, model_configs.model_hyperparameters.projected_size
        )

        self.dropout = nn.Dropout(model_configs.model_hyperparameters.projected_dropout)

        self._set_pooler_layer_to_trainable()

        self.loss_fn_type = model_configs.loss_function
        self.loss_fn = nn.MSELoss()

    @staticmethod
    def cosine_similarity_to_pkd(cosine_similarity, pkd_upper_bound, pkd_lower_bound):
        pkd_range = pkd_upper_bound - pkd_lower_bound
        return (cosine_similarity + 1) / 2 * pkd_range + pkd_lower_bound

    def forward(self, batch_input, **kwargs):
        """
        Forward pass of the model.

        Args:
            batch_input (dict): The input batch containing protein and drug input IDs and attention masks.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: The forward output containing cosine similarity, protein embedding, drug embedding, and loss (if labels are provided).
        """
        forward_output = {}

        protein_embedding = self.protein_model(
            input_ids=batch_input["protein_input_ids"],
            attention_mask=batch_input["protein_attention_mask"],
        )["pooler_output"]
        protein_embedding = self.dropout(protein_embedding)
        protein_embedding = self.protein_projection(protein_embedding)

        drug_embedding = self.drug_model(
            input_ids=batch_input["drug_input_ids"],
            attention_mask=batch_input["drug_attention_mask"],
        )["pooler_output"]
        drug_embedding = self.dropout(drug_embedding)
        drug_embedding = self.drug_projection(drug_embedding)

        if self.relu_before_cosine:
            protein_embedding = F.relu(protein_embedding)
            drug_embedding = F.relu(drug_embedding)

        cosine_similarity = F.cosine_similarity(protein_embedding, drug_embedding)
        if "labels" in batch_input:
            if batch_input["labels"] is not None:
                forward_output["loss"] = self.loss_fn(
                    cosine_similarity, batch_input["labels"]
                )

        forward_output["cosine_similarity"] = cosine_similarity
        forward_output["protein_embedding"] = protein_embedding
        forward_output["drug_embedding"] = drug_embedding

        return forward_output