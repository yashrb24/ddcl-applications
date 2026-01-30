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
                 act_embedding_table: nn.Embedding) -> None:
        super().__init__()
        # we are only passing one embedding table (for action tokens)
        # so we will have one embedding table, but two block masks
        # assert len(block_masks) == len(embedding_tables)
        assert ((act_mask + obs_mask) == 1).all()  # block mask are a partition of a block
        self.act_embedding_table = act_embedding_table
        self.embedding_dim = act_embedding_table.embedding_dim
        # assert all([e.embedding_dim == self.embedding_dim for e in embedding_tables]) # we have a single embedding table
        self.act_slicer, self.obs_slicer = Slicer(max_blocks, act_mask), Slicer(max_blocks, obs_mask)
        self.multipliers = None
        self.base = None
        self.delta = None
        self.uniform_dist = None

    def forward(self, tokenizer_output: TokenizerEncoderOutput, num_steps: int, prev_steps: int) -> torch.Tensor:
        tokens = tokenizer_output.tokens
        assert tokens.ndim == 2  # x is (B, T)
        output = torch.zeros(*tokens.size(), self.embedding_dim, device=tokens.device)
        # for slicer, emb in zip(self.slicers, self.embedding_tables):
        #     s = slicer.compute_slice(num_steps, prev_steps)
        #     output[:, s] = emb(tokens[:, s])
        act_slice = self.act_slicer.compute_slice(num_steps, prev_steps)
        output[:, act_slice] = self.act_embedding_table(tokens[:, act_slice])

        obs_slice = self.obs_slicer.compute_slice(num_steps, prev_steps)
        obs_tokens = tokens[:, obs_slice]
        m = self.token_to_message(obs_tokens)

        if self.delta is None:
            self.delta = tokenizer_output.delta
            self.uniform_dist = torch.distributions.Uniform(-self.delta / 2, self.delta / 2)

        epsilon = tokenizer_output.epsilon
        if tokenizer_output.epsilon is None:
            epsilon = self.uniform_dist.sample(m.shape)

        c_m = (m + 0.5) * self.delta
        z_hat = c_m - epsilon

        output[:, obs_slice] = z_hat

        return output

    def token_to_message(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.multipliers is None:
            powers = torch.arange(self.embedding_dim, device=tokens.device)
            self.base = 2 * self.scale + 1
            self.multipliers = torch.pow(self.base, powers)

        tokens = tokens.unsqueeze(-1)
        shifted_messages = (tokens // self.multipliers) % self.base.int()
        messages = shifted_messages - self.base
        return messages
