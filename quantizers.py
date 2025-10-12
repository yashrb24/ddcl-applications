import torch
import torch.nn as nn
from vector_quantize_pytorch import FSQ


class DDCL_Bottleneck(nn.Module):
    """
    A VAE bottleneck using the DDCL discretization procedure.
    This is a plug-and-play replacement for a VQ-VAE/FSQ codebook.
    """

    def __init__(self, delta, latent_dim=4):
        super().__init__()
        self.delta = delta
        self.latent_dim = latent_dim

    def forward(self, z):
        """
        The forward pass for TRAINING, using the reparameterization trick.
        """
        noise = (torch.rand_like(z) - 0.5) * self.delta
        z_prime = z + noise
        m = torch.floor(z_prime / self.delta).float() # float because of weights and biases are stored as floats, torch.floor converts them to integers

        comm_loss = torch.log2((2 * torch.abs(z) / self.delta) + 1).mean()

        return m, None, comm_loss  # Return None for indices to match VQ output

    # @torch.no_grad()
    # def quantize_and_dequantize(self, z):
    #     """
    #     The INFERENCE pass. This performs the full, non-differentiable
    #     quantization and de-quantization round trip.
    #     """
    #     # --- Sender Side ---
    #     noise = (torch.rand_like(z) - 0.5) * self.delta
    #     z_prime = z + noise
    #     indices = torch.floor(z_prime / self.delta).long()

    #     # --- Receiver Side ---
    #     C_m = self.delta * (indices.float() + 0.5)
    #     z_q = C_m - noise
    #     return z_q, indices


class FSQWrapper(nn.Module):
    """Wrapper around FSQ to match the interface of DDCL_Bottleneck"""

    def __init__(self, levels):
        super().__init__()
        self.fsq = FSQ(levels=levels)
        self.codebook_size = torch.prod(torch.tensor(levels)).item()

    def forward(self, z):
        """Forward pass that matches DDCL interface"""
        z_q, indices = self.fsq(z)
        return z_q, indices, 0.0  # No regularization loss for FSQ
