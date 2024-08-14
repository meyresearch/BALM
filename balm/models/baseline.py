import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

from balm.configs import ModelConfigs
from balm.models.base_model import BaseModel


class BaselineModel(BaseModel):
    """
    BaselineModel model extends BaseModel to concatenate protein and ligand encodings.
    This model takes the embeddings from both the protein and ligand models, concatenates them, and processes them further.

    Attributes:
        model_configs (ModelConfigs): The configuration object for the model.
        protein_model (AutoModel): The pre-trained protein model.
        drug_model (AutoModel): The pre-trained drug model.
        protein_embedding_size (int): The size of the protein model embeddings.
        drug_embedding_size (int): The size of the drug model embeddings.
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
        Forward pass for the BaselineModel.

        This method takes the input for both protein and drug models, obtains their embeddings, concatenates them, and processes them further.

        Args:
            protein_input (torch.Tensor): The input tensor for the protein model.
            drug_input (torch.Tensor): The input tensor for the drug model.

        Returns:
            torch.Tensor: The output tensor after processing the concatenated embeddings.
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
