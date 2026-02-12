import math

import torch
from torch import nn
from einops import rearrange


class DDCL(nn.Module):
    """
    DDCL stochastic quantizer with input/output projections and composite indexing.

    Projects the input from ``dim`` channels down to ``n_dims``, applies
    DDCL quantization, then projects back up.  Also provides composite
    (mixed-radix) indexing so that the ``n_dims`` per-element integers are
    packed into a single token id in ``[0, codebook_size)``.

    Parameters
    ----------
    n_dims : int
        Number of quantized dimensions (the bottleneck width).
    delta : float
        Quantization bin width.
    scale : float
        Tanh pre-scaling factor.  Latents are bounded to (-scale, +scale).
    dim : int or None
        Input channel dimension.  If None or equal to ``n_dims``, no
        projection is applied.
    ddcl_lambda : float
        Multiplier on the communication cost returned alongside the
        quantized output.
    """

    def __init__(
        self,
        n_dims: int = 4,
        delta: float = 1.0,
        scale: float = 3.5,
        dim: int | None = None,
        ddcl_lambda: float = 1e-3,
    ):
        super().__init__()
        self.n_dims = n_dims
        self.delta = delta
        self.scale = scale
        self.ddcl_lambda = ddcl_lambda

        need_proj = dim is not None and dim != n_dims
        self.project_in = nn.Linear(dim, n_dims) if need_proj else nn.Identity()
        self.project_out = nn.Linear(n_dims, dim) if need_proj else nn.Identity()

        half_delta = delta / 2.0
        min_m = math.floor((-scale - half_delta) / delta)
        max_m = math.floor((scale + half_delta) / delta)
        n_levels = max_m - min_m + 1
        codebook_size = n_levels**n_dims

        self.n_levels = n_levels
        self.min_m = min_m
        self.codebook_size = codebook_size

        offsets = [n_levels ** (n_dims - 1 - i) for i in range(n_dims)]
        self.register_buffer("_offsets", torch.tensor(offsets, dtype=torch.long))

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, fmap):
        """
        Parameters
        ----------
        fmap : Tensor (b, c, h, w)
            Encoder feature map.

        Returns
        -------
        fmap_out : Tensor (b, c, h, w)
            Quantized feature map (STE), same spatial size.
        indices : LongTensor (b, h, w)
            Single composite token index per spatial position.
        comm_loss : scalar Tensor
            Lambda-scaled communication cost.
        """

        z = rearrange(fmap, "b c h w -> b h w c")
        z = self.project_in(z)  # (b, h, w, n_dims)

        z = self.scale * torch.tanh(z)

        epsilon = (torch.rand_like(z) - 0.5) * self.delta

        z_prime = z + epsilon
        m = torch.floor(z_prime / self.delta)

        c_m = self.delta * (m + 0.5)
        z_hat = c_m - epsilon
        z_approx = z + (z_hat - z).detach()

        comm_loss = self.ddcl_lambda * torch.log2(z.abs() / self.delta + 1.0).mean()

        m_shifted = (m.long() - self.min_m).clamp(0, self.n_levels - 1)
        indices = (m_shifted * self._offsets).sum(dim=-1)

        fmap_out = self.project_out(z_approx)
        fmap_out = rearrange(fmap_out, "b h w c -> b c h w")

        return fmap_out, indices, comm_loss

    # ------------------------------------------------------------------
    # decode helper
    # ------------------------------------------------------------------

    def indices_to_codes(self, ids):
        dims = []
        remainder = ids
        for i in range(self.n_dims):
            dims.append(remainder // self._offsets[i])
            remainder = remainder % self._offsets[i]

        m_shifted = torch.stack(dims, dim=-1)
        m = m_shifted + self.min_m

        codes = self.delta * (m.float() + 0.5)

        fmap = self.project_out(codes)
        return rearrange(fmap, "b h w c -> b c h w")


# import torch
# from torch import nn


# class DDCL(nn.Module):
#     """
#     Implementation of the quantizer used in DDCL (Differentiable Discrete Communication Learning)
#     https://arxiv.org/pdf/2511.01554
#     """

#     def __init__(self, ddcl_delta: float, scale: float):
#         super().__init__()
#         self.delta = ddcl_delta
#         # self.ddcl_lambda = ddcl_lambda
#         self.scale = scale
#         self.tanh = nn.Tanh()
#         self.uniform_dist = torch.distributions.Uniform(-1, 1)

#     def forward(self, z):
#         # noise = (torch.rand_like(z) - 0.5) * 2 * self.ddcl_delta
#         # z_q = z + noise
#         # quantized = torch.floor(z_q / self.ddcl_delta).long()

#         # comm_loss = (
#         #     self.ddcl_lambda
#         #     * torch.log2((2 * torch.abs(z) / self.ddcl_delta) + 1).mean()
#         # )

#         # return z_q, quantized, comm_loss

#         z = self.scale * self.tanh(z)

#         epsilon = self.uniform_dist.sample(z.shape).to(z.device)
#         z_prime = z + epsilon

#         quantized = torch.floor(z_prime / self.delta)

#         c_m = self.delta * (quantized + 0.5)
#         z_hat = c_m - epsilon

#         e = (z_hat - z).detach()

#         z_approx = z + e

#         comm_loss = torch.log2((torch.abs(z) / self.delta) + 1).mean()

#         return z_approx, quantized, comm_loss
