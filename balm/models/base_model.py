import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel
from balm.configs import ModelConfigs


class BaseModel(nn.Module):
    """
    BaseModel integrates both protein and ligand laguage models.
    
    Attributes:
        model_configs (ModelConfigs): The configuration object for the model.
        protein_model (AutoModel): The pre-trained protein model.
        drug_model (AutoModel): The pre-trained ligand model.
        protein_embedding_size (int): The size of the protein model embeddings.
        drug_embedding_size (int): The size of the drug model embeddings.
    """

    def __init__(self, model_configs):
        """
        Initializes the BaseModel with the given configuration.

        Args:
            model_configs (ModelConfigs): The configuration object for the model.
        """
        super(BaseModel, self).__init__()
        self.model_configs = model_configs

        self.protein_model = AutoModel.from_pretrained(
            model_configs.protein_model_name_or_path,
            device_map="auto",
        )
        self.drug_model = AutoModel.from_pretrained(
            model_configs.drug_model_name_or_path,
            device_map="auto",
        )

        for name, param in self.protein_model.named_parameters():
            param.requires_grad = False

        for name, param in self.drug_model.named_parameters():
            param.requires_grad = False

        self._set_pooler_layer_to_trainable()

        self.protein_embedding_size = self.protein_model_embedding_size.get(
            model_configs.protein_model_name_or_path, protein_embedding_size
        )
        self.drug_embedding_size = self.drug_model_embedding_size.get(
            model_configs.drug_model_name_or_path, drug_embedding_size
        )

    def _set_pooler_layer_to_trainable(self):
        """
        Manually sets the pooler layer to be trainable for both protein and drug models.
        """
        for name, param in self.protein_model.named_parameters():
            if "pooler.dense" in name:
                param.requires_grad = True
        for name, param in self.drug_model.named_parameters():
            if "pooler.dense" in name:
                param.requires_grad = True

    def print_trainable_params(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for name, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            all_param += num_params
            if param.requires_grad:
                print(name)
                trainable_params += num_params

        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )