"""
Credits to https://github.com/CompVis/taming-transformers
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from dataset import Batch
from einops import rearrange
from typing import Any, Tuple

from sympy.abc import delta
from utils import LossWithIntermediateLosses

from .lpips import LPIPS
from .nets import Encoder, Decoder


@dataclass
class TokenizerEncoderOutput:
    z: torch.FloatTensor
    z_quantized: torch.FloatTensor
    tokens: torch.LongTensor
    z_scaled: dict


class Tokenizer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, encoder: Encoder, decoder: Decoder, scale: float, delta: float,
                 with_lpips: bool = True) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder = encoder
        self.pre_quant_conv = torch.nn.Conv2d(encoder.config.z_channels, embed_dim, 1)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, decoder.config.z_channels, 1)
        self.decoder = decoder
        self.scale = scale
        self.delta = delta

        self.embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)
        self.lpips = LPIPS().eval() if with_lpips else None
        self.tanh = nn.Tanh()
        self.uniform_dist = torch.distributions.Uniform(-delta / 2, delta / 2)

        # utils for token_to_message function
        self.multipliers = None

    def __repr__(self) -> str:
        return "tokenizer"

    def forward(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False):
        outputs = self.encode(x, should_preprocess)
        # decoder_input = outputs.z + (outputs.z_quantized - outputs.z).detach()
        decoder_input = outputs.z_quantized
        reconstructions = self.decode(decoder_input, should_postprocess)
        return outputs.z, outputs.z_quantized, reconstructions, outputs.z_scaled

    def compute_loss(self, batch: Batch, **kwargs: Any) -> LossWithIntermediateLosses:
        assert self.lpips is not None
        observations = self.preprocess_input(rearrange(batch['observations'], 'b t c h w -> (b t) c h w'))
        z, z_quantized, reconstructions, z_scaled = self(observations, should_preprocess=False, should_postprocess=False)

        """
        Old: VQ-VAE setup
        """
        # Codebook loss. Notes:
        # - beta position is different from taming and identical to original VQVAE paper
        # - VQVAE uses 0.25 by default
        # beta = 1.0
        # commitment_loss = (z.detach() - z_quantized).pow(2).mean() + beta * (z - z_quantized.detach()).pow(2).mean()

        """
        New: DDCL setup
        """
        # we don't have a commitment loss for ddcl, but we use this variable as a replacement for our ddcl loss
        commitment_loss = torch.log2(z_scaled / self.delta + 1).mean()

        reconstruction_loss = torch.abs(observations - reconstructions).mean()
        perceptual_loss = torch.mean(self.lpips(observations, reconstructions))

        return LossWithIntermediateLosses(commitment_loss=commitment_loss, reconstruction_loss=reconstruction_loss,
                                          perceptual_loss=perceptual_loss)

    def encode(self, x: torch.Tensor, should_preprocess: bool = False) -> TokenizerEncoderOutput:
        if should_preprocess:
            x = self.preprocess_input(x)
        shape = x.shape  # (..., C, H, W)
        x = x.view(-1, *shape[-3:])
        z = self.encoder(x)
        z = self.pre_quant_conv(z)
        # b, e, h, w = z.shape
        z_flattened = rearrange(z, 'b e h w -> (b h w) e')

        """
        Old: VQ-VAE setup
        """
        # dist_to_embeddings = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        # tokens = dist_to_embeddings.argmin(dim=-1)
        # z_q = rearrange(self.embedding(tokens), '(b h w) e -> b e h w', b=b, e=e, h=h, w=w).contiguous()
        # tokens = tokens.reshape(*shape[:-3], -1)

        # d1 d2 d3 d4 d5
        # delta = 20
        # scale * tanh(d1),scale * tanh(d2),  ...
        # (scale + delta/2)/delta , (- scale - delta2)/delta




        """
        New: DDCL setup
        """
        z_scaled = self.scale * self.tanh(z_flattened)
        epsilon = self.uniform_dist.sample(z_flattened.shape)
        z_prime = z_scaled + epsilon
        m = torch.floor(z_prime / self.delta)
        c_m = (m + 0.5) * self.delta
        z_hat = c_m - epsilon

        error = (z_hat - z_scaled).detach()
        z_q = z_scaled + error
        # Reshape to original
        z = z.reshape(*shape[:-3], *z.shape[1:])
        z_q = z_q.reshape(*shape[:-3], *z_q.shape[1:])
        tokens = self.message_to_token(m)
        tokens = tokens.reshape(*shape[:-3], -1)

        return TokenizerEncoderOutput(z, z_q, tokens, z_scaled)

    def decode(self, z_q: torch.Tensor, should_postprocess: bool = False) -> torch.Tensor:
        shape = z_q.shape  # (..., E, h, w)
        z_q = z_q.view(-1, *shape[-3:])
        z_q = self.post_quant_conv(z_q)
        rec = self.decoder(z_q)
        rec = rec.reshape(*shape[:-3], *rec.shape[1:])
        if should_postprocess:
            rec = self.postprocess_output(rec)
        return rec

    @torch.no_grad()
    def encode_decode(self, x: torch.Tensor, should_preprocess: bool = False,
                      should_postprocess: bool = False) -> torch.Tensor:
        z_q = self.encode(x, should_preprocess).z_quantized
        return self.decode(z_q, should_postprocess)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """x is supposed to be channels first and in [0, 1]"""
        return x.mul(2).sub(1)

    def postprocess_output(self, y: torch.Tensor) -> torch.Tensor:
        """y is supposed to be channels first and in [-1, 1]"""
        return y.add(1).div(2)

    def message_to_token(self, message: torch.Tensor) -> torch.Tensor:
        if self.multipliers is None:
            d = message.shape[-1]
            powers = torch.arange(d, device=message.device)
            self.multipliers = torch.pow(self.scale, powers)

        tokens = torch.linalg.vecdot(self.multipliers, message)
        return tokens

    def token_to_message(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.unsqueeze(-1)
        messages = ((tokens // self.multipliers) % self.scale).int()
        return messages
