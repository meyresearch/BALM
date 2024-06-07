import torch
from torch import nn
from torch.nn import functional as F


class EuclideanMAELoss(nn.Module):
    def __init__(self):
        super(EuclideanMAELoss, self).__init__()

    def forward(self, x_1, x_2, labels):
        # Calculate Euclidean distance between x_1 and x_2
        euclidean_distance = torch.norm(x_1 - x_2, dim=1)

        # Calculate Mean Absolute Error (MAE) between euclidean_distance and labels
        mae_loss = F.l1_loss(euclidean_distance, labels)

        return mae_loss
