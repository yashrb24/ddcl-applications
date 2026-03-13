"""
Credits to https://github.com/CompVis/taming-transformers
"""

import math

import torch
import torch.nn as nn
import numpy as np
from vector_quantize_pytorch import FSQ
from dataclasses import dataclass
from dataset import Batch
from einops import rearrange
from typing import Any, List, Optional, Tuple

from utils import LossWithIntermediateLosses

from .lpips import LPIPS
from .nets import Encoder, Decoder


@dataclass
class TokenizerEncoderOutput:
    z: torch.FloatTensor
    z_quantized: torch.FloatTensor
    z_scaled: torch.FloatTensor
    tokens: torch.LongTensor
    epsilon: torch.FloatTensor


class Tokenizer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, encoder: Encoder, decoder: Decoder, scale: float, delta: float,
                 enable_ddcl: bool = True, enable_fsq: bool = False, fsq_levels: Optional[List[int]] = None,
                 with_lpips: bool = True) -> None:
        super().__init__()
        assert not (enable_ddcl and enable_fsq), "enable_ddcl and enable_fsq cannot both be True"
        self.enable_ddcl = enable_ddcl
        self.enable_fsq = enable_fsq
        self.encoder = encoder
        self.pre_quant_conv = torch.nn.Conv2d(encoder.config.z_channels, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, decoder.config.z_channels, 1)
        self.decoder = decoder
        self.scale = scale
        self.delta = delta

        if enable_fsq:
            self.fsq_levels = fsq_levels if fsq_levels is not None else [8, 8, 8, 8]
            self.vocab_size = math.prod(self.fsq_levels)
            self.fsq = FSQ(levels=self.fsq_levels, dim=embed_dim, channel_first=True, return_indices=True)
            self.embedding = nn.Embedding(self.vocab_size, embed_dim)
            self.embedding.weight.data.uniform_(-1.0 / self.vocab_size, 1.0 / self.vocab_size)
        else:
            self.vocab_size = vocab_size
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)

        self.lpips = LPIPS().eval() if with_lpips else None

        if enable_ddcl:
            half_delta = delta / 2.0
            self.min_m = math.floor((-scale - half_delta) / delta)
            max_m = math.floor((scale + half_delta) / delta)
            self.n_levels = max_m - self.min_m + 1
            self.tanh = nn.Tanh()
            self.uniform_dist = torch.distributions.Uniform(-delta / 2, delta / 2)
            self.multipliers = None

    def __repr__(self) -> str:
        return "tokenizer"

    def forward(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False):
        outputs = self.encode(x, should_preprocess)
        if self.enable_ddcl:
            decoder_input = outputs.z_quantized
        else:
            # Straight-through estimator for both VQVAE and FSQ
            decoder_input = outputs.z + (outputs.z_quantized - outputs.z).detach()
        reconstructions = self.decode(decoder_input, should_postprocess)
        return outputs.z, outputs.z_quantized, reconstructions, outputs.z_scaled

    def compute_loss(self, batch: Batch, **kwargs: Any) -> LossWithIntermediateLosses:
        assert self.lpips is not None
        observations = self.preprocess_input(rearrange(batch['observations'], 'b t c h w -> (b t) c h w'))
        z, z_quantized, reconstructions, z_scaled = self(observations, should_preprocess=False,
                                                         should_postprocess=False)

        if self.enable_fsq:
            commitment_loss = torch.tensor(0.0, device=z.device)
        elif self.enable_ddcl:
            commitment_loss = torch.log2(z_scaled / self.delta + 1).mean()
        else:
            beta = 1.0
            commitment_loss = (z.detach() - z_quantized).pow(2).mean() + beta * (z - z_quantized.detach()).pow(2).mean()

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
        b, e, h, w = z.shape
        z_flattened = rearrange(z, 'b e h w -> (b h w) e')

        if self.enable_fsq:
            z_q, indices = self.fsq(z)  # z is already (b, e, h, w)
            tokens = indices.reshape(b, -1)
            z = z.reshape(*shape[:-3], *z.shape[1:])
            z_q = z_q.reshape(*shape[:-3], *z_q.shape[1:])
            tokens = tokens.reshape(*shape[:-3], -1)
            return TokenizerEncoderOutput(z=z, z_quantized=z_q, z_scaled=None, tokens=tokens, epsilon=None)
        elif self.enable_ddcl:
            z_scaled = self.scale * self.tanh(z_flattened)
            epsilon = self.uniform_dist.sample(z_flattened.shape).to(device=z_flattened.device, dtype=z_flattened.dtype)
            z_prime = z_scaled + epsilon
            m = torch.floor(z_prime / self.delta)
            c_m = (m + 0.5) * self.delta
            z_hat = c_m - epsilon

            error = (z_hat - z_scaled).detach()
            z_q = z_scaled + error
            z_q = rearrange(z_q, '(b h w) e -> b e h w', b=b, h=h, w=w)
            z_scaled = rearrange(z_scaled, '(b h w) e -> b e h w', b=b, h=h, w=w)
            z = z.reshape(*shape[:-3], *z.shape[1:])
            z_q = z_q.reshape(*shape[:-3], *z_q.shape[1:])
            z_scaled = z_scaled.reshape(*shape[:-3], *z_scaled.shape[1:])
            tokens = self.message_to_token(m)
            tokens = tokens.reshape(*shape[:-3], -1)
            return TokenizerEncoderOutput(z=z, z_quantized=z_q, z_scaled=z_scaled, tokens=tokens, epsilon=epsilon)
        else:
            dist_to_embeddings = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight ** 2, dim=1) - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
            tokens = dist_to_embeddings.argmin(dim=-1)
            z_q = rearrange(self.embedding(tokens), '(b h w) e -> b e h w', b=b, e=e, h=h, w=w).contiguous()
            z = z.reshape(*shape[:-3], *z.shape[1:])
            z_q = z_q.reshape(*shape[:-3], *z_q.shape[1:])
            tokens = tokens.reshape(*shape[:-3], -1)
            return TokenizerEncoderOutput(z=z, z_quantized=z_q, z_scaled=None, tokens=tokens, epsilon=None)

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
            self.multipliers = torch.pow(self.n_levels, powers).float()

        shifted_message = message - self.min_m
        shifted_message = shifted_message.clamp(0, self.n_levels - 1)
        tokens = torch.linalg.vecdot(self.multipliers, shifted_message)
        return tokens

    def decode_from_tokens(self, tokens: torch.LongTensor) -> torch.Tensor:
        """Decode observation tokens back to pixel space. Handles VQVAE, DDCL, and FSQ modes.

        Args:
            tokens: (B, K) token indices where K = h*w

        Returns:
            Reconstructed observations (B, C, H, W) in [0, 1]
        """
        h = int(np.sqrt(tokens.shape[1]))

        if self.enable_fsq:
            # Reshape to spatial before indices_to_codes so channel_first rearrangement works
            indices_2d = rearrange(tokens, 'b (h w) -> b h w', h=h)
            z = self.fsq.indices_to_codes(indices_2d)  # (B, embed_dim, h, w) with channel_first=True
        else:
            embedded = self.embedding(tokens)  # (B, K, E)
            z = rearrange(embedded, 'b (h w) e -> b e h w', h=h)

        rec = self.decode(z, should_postprocess=True)
        return torch.clamp(rec, 0, 1)
