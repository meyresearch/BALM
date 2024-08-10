import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

from balm.configs import ModelConfigs
from balm.models.base_model import BaseModel


class BaselineModel(BaseModel):
    """
    Baseline model for protein-drug interaction prediction.

    Args:
        model_configs (ModelConfigs): Configuration object for the model.
        protein_embedding_size (int, optional): Size of the protein embedding. Defaults to 640.
        drug_embedding_size (int, optional): Size of the drug embedding. Defaults to 384.
    """

    def __init__(
        self,
        model_configs: ModelConfigs,
        protein_embedding_size=640,
        drug_embedding_size=384,
    ):
        super(BaselineModel, self).__init__(
            model_configs, protein_embedding_size, drug_embedding_size
        )

        # combined layers
        self.linear_projection = nn.Linear(
            self.protein_embedding_size + self.drug_embedding_size,
            model_configs.model_hyperparameters.projected_size,
        )
        self.dropout = nn.Dropout(model_configs.model_hyperparameters.projected_dropout)
        self.out = nn.Linear(model_configs.model_hyperparameters.projected_size, 1)

        self.print_trainable_params()

        self.loss_fn = nn.MSELoss()

    def forward(self, batch_input, **kwargs):
        """
        Forward pass of the model.

        Args:
            batch_input (dict): Input batch containing protein and drug input IDs and attention masks.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: Dictionary containing the forward output, including protein and drug embeddings, and logits.
        """
        forward_output = {}

        protein_embedding = self.protein_model(
            input_ids=batch_input["protein_input_ids"],
            attention_mask=batch_input["protein_attention_mask"],
        )["pooler_output"]

        drug_embedding = self.drug_model(
            input_ids=batch_input["drug_input_ids"],
            attention_mask=batch_input["drug_attention_mask"],
        )["pooler_output"]

        # concat
        concatenated_embedding = torch.cat((protein_embedding, drug_embedding), 1)

        # add some dense layers
        projected_embedding = F.relu(self.linear_projection(concatenated_embedding))
        projected_embedding = self.dropout(projected_embedding)
        logits = self.out(projected_embedding)

        if batch_input["labels"] is not None:
            forward_output["loss"] = self.loss_fn(logits, batch_input["labels"])

        forward_output["protein_embedding"] = protein_embedding
        forward_output["drug_embedding"] = drug_embedding
        forward_output["logits"] = logits.squeeze(-1)

        return forward_output
