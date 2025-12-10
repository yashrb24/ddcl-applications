"""
Quantization methods for the tokenizer.
"""

from abc import ABC, abstractmethod
from typing import Tuple

from einops import rearrange
import torch
import torch.nn as nn


class Quantizer(ABC, nn.Module):
    """Base class for quantization methods."""

    @abstractmethod
    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        Quantize continuous latent representations.

        Args:
            z: Continuous latent tensor of shape (b, e, h, w)

        Returns:
            z_quantized: Quantized latent tensor of shape (b, e, h, w)
            tokens: Discrete token indices of shape (b*h*w,)
        """
        pass

    @abstractmethod
    def compute_commitment_loss(self, z: torch.Tensor, z_quantized: torch.Tensor) -> torch.Tensor:
        """
        Compute quantization-specific commitment loss.

        Args:
            z: Original continuous latents
            z_quantized: Quantized latents

        Returns:
            Commitment loss scalar
        """
        pass

    @abstractmethod
    def lookup_tokens(self, tokens: torch.LongTensor) -> torch.Tensor:
        """
        Convert discrete tokens back to continuous embeddings.

        This is the inverse of quantization - given token indices, return the
        corresponding continuous representations. For VQ this is a codebook lookup

        Args:
            tokens: Discrete token indices

        Returns:
            Continuous embeddings corresponding to the tokens
        """
        pass


class VQQuantizer(Quantizer):

    """Vector Quantization using L2 distance to nearest codebook vector."""

    def __init__(self, vocab_size: int, embed_dim: int):
        """
        Initialize VQ quantizer.

        Args:
            vocab_size: Size of the codebook (number of discrete tokens)
            embed_dim: Dimension of each codebook vector
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self._embedding = nn.Embedding(vocab_size, embed_dim)
        # Initialize codebook uniformly as in original implementation
        self._embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)

    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        Vector quantization: find nearest codebook vector for each spatial position.

        Args:
            z: Continuous latent tensor of shape (b, e, h, w)

        Returns:
            z_q: Quantized latent tensor of shape (b, e, h, w)
            tokens: Discrete token indices of shape (b*h*w,)
        """
        b, e, h, w = z.shape

        # Flatten spatial dimensions
        z_flattened = rearrange(z, 'b e h w -> (b h w) e')

        # Compute L2 distances to all codebook embeddings
        # dist = ||z||^2 + ||embeddings||^2 - 2 * z * embeddings^T
        dist_to_embeddings = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flattened, self._embedding.weight.t())
        )

        # Find nearest codebook vector
        tokens = dist_to_embeddings.argmin(dim=-1)

        # Look up quantized vectors and reshape
        z_q = rearrange(
            self._embedding(tokens),
            '(b h w) e -> b e h w',
            b=b, e=e, h=h, w=w
        ).contiguous()

        return z_q, tokens

    def compute_commitment_loss(self, z: torch.Tensor, z_quantized: torch.Tensor) -> torch.Tensor:
        """
        Compute VQ commitment loss (codebook + commitment).

        This uses the formulation from the original VQVAE paper with beta=1.0
        (different from typical taming-transformers which uses 0.25).

        Loss = ||sg[z] - z_q||^2 + beta * ||z - sg[z_q]||^2
        where sg[] is the stop-gradient operator.

        Args:
            z: Original continuous latents
            z_quantized: Quantized latents

        Returns:
            Commitment loss scalar
        """
        beta = 1.0
        commitment_loss = (
            (z.detach() - z_quantized).pow(2).mean()
            + beta * (z - z_quantized.detach()).pow(2).mean()
        )
        return commitment_loss

    def lookup_tokens(self, tokens: torch.LongTensor) -> torch.Tensor:
        """
        Look up codebook embeddings for the given tokens.

        Args:
            tokens: Discrete token indices

        Returns:
            Continuous embeddings from the codebook
        """
        return self._embedding(tokens)

class DDCLQuantizer(Quantizer):

    def __init__(self, delta: float, scale: float):
        super().__init__()
        self.delta = delta
        self.scale = scale
        
        # used in quantize method
        self.tanh = nn.Tanh()
        self.uniform_dist = torch.distributions.Uniform(-delta / 2, delta / 2)
        self.basis = None

    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Quantize continuous latent representations using DDCL."""
        noise = self.uniform_dist.sample(z.shape).to(z.device)
        z_prime = self.scale * self.tanh(z) + noise
        c_m = torch.floor(z_prime / self.delta)
        z_quantized = c_m - noise

        if self.basis is None:
            exponents = torch.arange(0, z.shape[-1], 1, device=z.device) 
            self.basis = self.scale ** exponents # (scale, scale^2, scale^3, ...)

        tokens = torch.sum(c_m * self.basis, dim=-1) # (d1 * scale + d2 * scale^2 + d3 * scale^3 + ...)

        return z_quantized, tokens.long()
        

    def compute_commitment_loss(self, z: torch.Tensor, z_quantized: torch.Tensor) -> torch.Tensor:
        return torch.log2(2 * z / self.delta + 1).mean()

    def lookup_tokens(self, tokens: torch.LongTensor) -> torch.Tensor:
        if self.basis is None:
            raise ValueError("Basis not initialized. Perform quantization first.")

        # convert the shape to (..., num_tokens, 1)
        tokens = tokens.unsqueeze(-1) 

        z_quantized = (tokens // self.basis) % self.scale

        return z_quantized