import torch
from torch import nn


class DDCL(nn.Module):
    """
    Implementation of the quantizer used in DDCL (Differentiable Discrete Communication Learning)
    https://arxiv.org/pdf/2511.01554
    """

    def __init__(self, ddcl_delta: float, scale: float):
        super().__init__()
        self.delta = ddcl_delta
        # self.ddcl_lambda = ddcl_lambda
        self.scale = scale
        self.tanh = nn.Tanh()
        self.uniform_dist = torch.distributions.Uniform(-1, 1)

    def forward(self, z):
        # noise = (torch.rand_like(z) - 0.5) * 2 * self.ddcl_delta
        # z_q = z + noise
        # quantized = torch.floor(z_q / self.ddcl_delta).long()

        # comm_loss = (
        #     self.ddcl_lambda
        #     * torch.log2((2 * torch.abs(z) / self.ddcl_delta) + 1).mean()
        # )

        # return z_q, quantized, comm_loss

        z = self.scale * self.tanh(z) 

        epsilon = self.uniform_dist.sample(z.shape).to(z.device)
        z_prime = z + epsilon

        quantized = torch.floor(z_prime / self.delta) 

        c_m = self.delta * (quantized + 0.5)
        z_hat = c_m - epsilon

        e = (z_hat - z).detach() 

        z_approx = z + e

        comm_loss = torch.log2((torch.abs(z) / self.delta) + 1).mean()

        return z_approx, quantized, comm_loss
