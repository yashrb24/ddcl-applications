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
        The forward pass for TRAINING and INFERENCE
        """
        noise = (torch.rand_like(z) - 0.5) * self.delta
        z_q = z + noise

        comm_loss = torch.log2((2 * torch.abs(z) / self.delta) + 1).mean()

        return z_q, None, comm_loss  # Return None for indices to match VQ output


class FSQWrapper(nn.Module):
    """Wrapper around FSQ to match the interface of DDCL_Bottleneck"""

    def __init__(self, levels):
        super().__init__()
        self.fsq = FSQ(levels=levels, channel_first=True)
        self.codebook_size = torch.prod(torch.tensor(levels)).item()

    def forward(self, z):
        """Forward pass that matches DDCL interface"""
        z_q, indices = self.fsq(z)
        return z_q, indices, 0.0  # No regularization loss for FSQ


class VanillaVAE(nn.Module):
    """
    Standard Variational Autoencoder bottleneck with Gaussian latent space.
    Uses reparameterization trick for sampling during training.
    """

    def __init__(self, latent_dim=4):
        super().__init__()
        self.latent_dim = latent_dim

    def forward(self, z):
        """
        Forward pass for VAE bottleneck.
        
        Args:
            z: Encoder output of shape (batch, 2*latent_dim) containing mu and logvar
            
        Returns:
            z_sampled: Sampled latent vector (batch, latent_dim)
            None: No indices for VAE
            kl_loss: KL divergence acts as regularization loss
        """
        # Split into mu and logvar
        mu = z[:, :self.latent_dim]
        logvar = z[:, self.latent_dim:]

        # Reparameterization trick: z = mu + sigma * epsilon
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z_sampled = mu + std * epsilon

        # KL divergence loss: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        return z_sampled, None, kl_loss
