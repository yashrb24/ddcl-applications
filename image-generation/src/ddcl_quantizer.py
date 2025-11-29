import torch
from torch import nn


class DDCL(nn.Module):
    """
    Implementation of the quantizer used in DDCL (Differentiable Discrete Communication Learning)
    https://arxiv.org/pdf/2511.01554
    """

    def __init__(self, ddcl_delta: float, ddcl_lambda: float):
        super().__init__()
        self.ddcl_delta = ddcl_delta
        self.ddcl_lambda = ddcl_lambda

    def forward(self, z):
        noise = (torch.rand_like(z) - 0.5) * 2 * self.ddcl_delta
        z_q = z + noise
        quantized = torch.floor(z_q / self.ddcl_delta).long()

        comm_loss = (
            self.ddcl_lambda
            * torch.log2((2 * torch.abs(z) / self.ddcl_delta) + 1).mean()
        )

        return z_q, quantized, comm_loss
