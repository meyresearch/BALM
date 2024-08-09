import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss


class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma, device):
        super(BMCLoss, self).__init__()
        self.device = device
        self.noise_sigma = torch.nn.Parameter(
            torch.tensor(init_noise_sigma, device=device)
        )

    def bmc_loss(self, pred, target, noise_var):
        """Compute the Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
        Args:
        pred: A float tensor of size [batch, 1].
        target: A float tensor of size [batch, 1].
        noise_var: A float number or tensor.
        Returns:
        loss: A float tensor. Balanced MSE Loss.
        """
        logits = -(pred - target.T).pow(2) / (
            2 * noise_var
        )  # logit size: [batch, batch]
        temp = torch.arange(pred.shape[0], dtype=torch.float32, device=self.device)
        loss = F.cross_entropy(logits, temp)  # contrastive-like loss
        loss = (
            loss * (2 * noise_var).detach()
        )  # optional: restore the loss scale, 'detach' when noise is learnable

        return loss

    def forward(self, pred, target):
        noise_var = self.noise_sigma**2
        return self.bmc_loss(pred, target, noise_var)
