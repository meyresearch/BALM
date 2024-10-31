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
    BALM model using protein and ligand language models, projecting them into a shared space 
    using a cosine similarity metric for compatibility.

    Args:
        model_configs (ModelConfigs): Configuration object for the model, containing all necessary hyperparameters.
        protein_embedding_size (int): Size of the protein embedding (default=640).
        drug_embedding_size (int): Size of the drug embedding (default=384).
    """

    def __init__(
        self,
        model_configs: ModelConfigs,
        protein_embedding_size=640,
        drug_embedding_size=384,
    ):
        # Initialize the base model with configurations and embedding sizes
        super(BALM, self).__init__(
            model_configs, protein_embedding_size, drug_embedding_size
        )

        # Set ReLU activation before cosine similarity if specified in the model configs
        self.relu_before_cosine = (
            self.model_configs.model_hyperparameters.relu_before_cosine
        )

        # Apply Protein PEFT (Parameter Efficient Fine-Tuning) configuration if provided
        if model_configs.protein_peft_hyperparameters:
            # Retrieve and apply protein PEFT configurations
            self.protein_peft_config = get_peft_config(
                model_configs.protein_fine_tuning_type
            )(
                **model_configs.protein_peft_hyperparameters,
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.protein_model = get_peft_model(
                self.protein_model, self.protein_peft_config
            )
            # Print trainable parameters in the protein model
            self.protein_model.print_trainable_parameters()

        # Apply Drug PEFT configuration if specified in model configs
        if model_configs.drug_peft_hyperparameters:
            # Retrieve and apply drug PEFT configurations
            self.drug_peft_config = get_peft_config(
                model_configs.drug_fine_tuning_type
            )(
                **model_configs.drug_peft_hyperparameters,
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.drug_model = get_peft_model(self.drug_model, self.drug_peft_config)
            # Print trainable parameters in the drug model
            self.drug_model.print_trainable_parameters()

        # Define linear projection layers for protein and drug embeddings
        self.protein_projection = nn.Linear(
            self.protein_embedding_size,
            model_configs.model_hyperparameters.projected_size,
        )
        self.drug_projection = nn.Linear(
            self.drug_embedding_size, model_configs.model_hyperparameters.projected_size
        )

        # Apply dropout with a rate defined in model configs
        self.dropout = nn.Dropout(model_configs.model_hyperparameters.projected_dropout)

        # Initialize pooler layers to be trainable
        self._set_pooler_layer_to_trainable()

        # Define loss function and type
        self.loss_fn_type = model_configs.loss_function
        self.loss_fn = nn.MSELoss()

    @staticmethod
    def cosine_similarity_to_pkd(cosine_similarity, pkd_upper_bound, pkd_lower_bound):
        """
        Converts cosine similarity scores to pKd values by scaling within defined bounds.

        Args:
            cosine_similarity (Tensor): Cosine similarity score(s) between protein and drug embeddings.
            pkd_upper_bound (float): Maximum pKd value.
            pkd_lower_bound (float): Minimum pKd value.

        Returns:
            Tensor: Scaled pKd values.
        """
        pkd_range = pkd_upper_bound - pkd_lower_bound
        return (cosine_similarity + 1) / 2 * pkd_range + pkd_lower_bound

    def forward(self, batch_input, **kwargs):
        """
        Forward pass of the BALM model.

        Args:
            batch_input (dict): Input batch containing protein and drug input IDs and attention masks.
            **kwargs: Additional keyword arguments for flexibility.

        Returns:
            dict: Output dictionary containing cosine similarity, embeddings, and optional loss.
        """
        forward_output = {}

        # Extract protein embeddings by passing input IDs and attention masks through the protein model
        protein_embedding = self.protein_model(
            input_ids=batch_input["protein_input_ids"],
            attention_mask=batch_input["protein_attention_mask"],
        )["pooler_output"]
        # Apply dropout to the protein embedding and project it to shared space
        protein_embedding = self.dropout(protein_embedding)
        protein_embedding = self.protein_projection(protein_embedding)

        # Extract drug embeddings in the same way
        drug_embedding = self.drug_model(
            input_ids=batch_input["drug_input_ids"],
            attention_mask=batch_input["drug_attention_mask"],
        )["pooler_output"]
        # Apply dropout to the drug embedding and project it to shared space
        drug_embedding = self.dropout(drug_embedding)
        drug_embedding = self.drug_projection(drug_embedding)

        # Optionally apply ReLU activation to both embeddings before cosine similarity
        if self.relu_before_cosine:
            protein_embedding = F.relu(protein_embedding)
            drug_embedding = F.relu(drug_embedding)

        # Compute cosine similarity between protein and drug embeddings
        cosine_similarity = F.cosine_similarity(protein_embedding, drug_embedding)

        # Calculate loss if labels are provided in the batch input
        if "labels" in batch_input:
            if batch_input["labels"] is not None:
                forward_output["loss"] = self.loss_fn(
                    cosine_similarity, batch_input["labels"]
                )

        # Store outputs in the dictionary
        forward_output["cosine_similarity"] = cosine_similarity
        forward_output["protein_embedding"] = protein_embedding
        forward_output["drug_embedding"] = drug_embedding

        return forward_output