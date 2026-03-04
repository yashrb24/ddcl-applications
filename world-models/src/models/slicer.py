import math

import torch
import torch.nn as nn

from src.models.tokenizer import TokenizerEncoderOutput


class Slicer(nn.Module):
    def __init__(self, max_blocks: int, block_mask: torch.Tensor) -> None:
        super().__init__()
        self.block_size = block_mask.size(0)
        self.num_kept_tokens = block_mask.sum().long().item()
        kept_indices = torch.where(block_mask)[0].repeat(max_blocks)
        offsets = torch.arange(max_blocks).repeat_interleave(self.num_kept_tokens)
        self.register_buffer('indices', kept_indices + block_mask.size(0) * offsets)

    def compute_slice(self, num_steps: int, prev_steps: int = 0) -> torch.Tensor:
        total_steps = num_steps + prev_steps
        num_blocks = math.ceil(total_steps / self.block_size)
        indices = self.indices[:num_blocks * self.num_kept_tokens]
        return indices[torch.logical_and(prev_steps <= indices, indices < total_steps)] - prev_steps

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Head(Slicer):
    def __init__(self, max_blocks: int, block_mask: torch.Tensor, head_module: nn.Module) -> None:
        super().__init__(max_blocks, block_mask)
        assert isinstance(head_module, nn.Module)
        self.head_module = head_module

    def forward(self, x: torch.Tensor, num_steps: int, prev_steps: int) -> torch.Tensor:
        x_sliced = x[:, self.compute_slice(num_steps, prev_steps)]  # x is (B, T, E)
        return self.head_module(x_sliced)


class Embedder(nn.Module):
    def __init__(self, max_blocks: int, act_mask: torch.Tensor, obs_mask: torch.Tensor,
                 act_embedding_table: nn.Embedding, enable_ddcl: bool, enable_fsq: bool = False,
                 scale: float = None, delta: float = None,
                 obs_embedding_table: nn.Embedding = None) -> None:
        super().__init__()
        assert ((act_mask + obs_mask) == 1).all()
        self.enable_ddcl = enable_ddcl
        self.enable_fsq = enable_fsq
        self.act_embedding_table = act_embedding_table
        self.embedding_dim = act_embedding_table.embedding_dim
        self.act_slicer, self.obs_slicer = Slicer(max_blocks, act_mask), Slicer(max_blocks, obs_mask)

        if enable_ddcl:
            self.scale = scale
            self.delta = delta
            self.num_levels = int(scale / delta)
            self.uniform_dist = torch.distributions.Uniform(-delta / 2, delta / 2)
            self.multipliers = None
        else:
            # Both VQVAE and FSQ use embedding table lookup
            self.obs_embedding_table = obs_embedding_table

    def forward(self, tokenizer_output: TokenizerEncoderOutput, num_steps: int, prev_steps: int) -> torch.Tensor:
        tokens = tokenizer_output.tokens
        assert tokens.ndim == 2  # x is (B, T)
        output = torch.zeros(*tokens.size(), self.embedding_dim, device=tokens.device)
        act_slice = self.act_slicer.compute_slice(num_steps, prev_steps)
        output[:, act_slice] = self.act_embedding_table(tokens[:, act_slice])

        obs_slice = self.obs_slicer.compute_slice(num_steps, prev_steps)

        if self.enable_ddcl:
            obs_tokens = tokens[:, obs_slice]
            m = self.token_to_message(obs_tokens)

            epsilon = tokenizer_output.epsilon
            if tokenizer_output.epsilon is None:
                epsilon = self.uniform_dist.sample(m.shape).to(device=m.device, dtype=m.dtype)

            c_m = (m + 0.5) * self.delta
            z_hat = c_m - epsilon
            output[:, obs_slice] = z_hat
        else:
            output[:, obs_slice] = self.obs_embedding_table(tokens[:, obs_slice])

        return output

    def token_to_message(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.multipliers is None:
            powers = torch.arange(self.embedding_dim, device=tokens.device)
            self.multipliers = torch.pow(2 * self.num_levels + 2, powers)

        tokens = tokens.unsqueeze(-1)
        shifted_messages = (tokens // self.multipliers) % (2 * self.num_levels + 2)
        messages = shifted_messages - self.num_levels - 1
        return messages
