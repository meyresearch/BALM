import torch
from info_nce import InfoNCE
from peft import (
    IA3Config,
    LoHaConfig,
    LoKrConfig,
    LoraConfig,
    PeftConfig,
    TaskType,
    get_peft_model,
)
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

from .configs import ModelConfigs

PEFT_TYPE_TO_CONFIG_MAPPING = {
    "lora": LoraConfig,
    "loha": LoHaConfig,
    "lokr": LoKrConfig,
    "ia3": IA3Config,
}


def get_peft_config(peft_type):
    return PEFT_TYPE_TO_CONFIG_MAPPING[peft_type]


class BaseModel(nn.Module):
    protein_model_embedding_size = {
        "facebook/esm2_t30_150M_UR50D": 640,
        "facebook/esm2_t33_650M_UR50D": 1280,
    }
    drug_model_embedding_size = {"DeepChem/ChemBERTa-77M-MTR": 384}

    def __init__(
        self,
        model_configs: ModelConfigs,
        protein_embedding_size: int,
        drug_embedding_size: int,
    ):
        """
        Initializes the BaseModel.

        Args:
            model_configs (ModelConfigs): The configuration object for the model.
        """
        super(BaseModel, self).__init__()

        self.protein_model = AutoModel.from_pretrained(
            model_configs.protein_model_name_or_path
        )
        self.drug_model = AutoModel.from_pretrained(
            model_configs.drug_model_name_or_path
        )

        # Manually set pooler layer to be trainable
        for name, param in self.protein_model.named_parameters():
            if "pooler.dense" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        for name, param in self.drug_model.named_parameters():
            if "pooler.dense" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.protein_embedding_size = self.protein_model_embedding_size.get(
            model_configs.protein_model_name_or_path, protein_embedding_size
        )
        self.drug_embedding_size = self.drug_model_embedding_size.get(
            model_configs.drug_model_name_or_path, drug_embedding_size
        )


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

        if kwargs["labels"] is not None:
            forward_output["loss"] = self.loss_fn(logits, kwargs["labels"])

        forward_output["protein_embedding"] = protein_embedding
        forward_output["drug_embedding"] = drug_embedding
        forward_output["logits"] = logits.squeeze(-1)

        return forward_output


class BindingAffinityModel(BaseModel):
    """
    A class representing the Binding Affinity Model.

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
        super(BindingAffinityModel, self).__init__(
            model_configs, protein_embedding_size, drug_embedding_size
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

        self.loss_fn_type = model_configs.loss_function
        self.loss_fn_hyperparameters = model_configs.loss_hyperparameters
        self.loss_fn = self._set_loss()

    def _set_loss(self):
        """
        Set the loss function based on the loss function type specified in the model configurations.

        Returns:
            nn.Module: The loss function module.
        """
        if self.loss_fn_type == "cosine_mse":
            return nn.MSELoss()

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

        cosine_similarity = F.cosine_similarity(protein_embedding, drug_embedding)
        if kwargs["labels"] is not None:
            forward_output["loss"] = self.loss_fn(cosine_similarity, kwargs["labels"])

        forward_output["cosine_similarity"] = cosine_similarity
        forward_output["protein_embedding"] = protein_embedding
        forward_output["drug_embedding"] = drug_embedding

        return forward_output
