# Agents Guide — image-generation

## Project Overview

This project benchmarks **Differentiable Discrete Communication Learning (DDCL)** ([arxiv.org/abs/2511.01554](https://arxiv.org/abs/2511.01554)) against VQ and FSQ quantization methods for image generation. It implements a two-stage pipeline: (1) train a VQ-GAN tokenizer with one of three quantizers, (2) train a MaskGit transformer for class-conditional image generation.

Supported datasets:
- **ImageNet 256×256** — 1000-class, 1.2M images, 4-layer VAE (16× downsample)
- **CIFAR-10 128×128** — 10-class, 50k images (pre-upscaled), 3-layer VAE (8× downsample)

Both produce 16×16 latent maps (256 tokens) for fair comparison.

---

## Repository Structure

```
image-generation/
├── AGENTS.md                          # This file
├── pyproject.toml                     # Python ≥3.12, deps (torch, accelerate, etc.)
├── README.md                          # Full usage guide
├── scripts/
│   ├── download_imagenet.py           # Data download: HuggingFace → ImageFolder
│   ├── imagenet/                      # ImageNet 256×256 training entry points
│   │   ├── train_vae_vq.py            # Stage I: VQ tokenizer (layers=4)
│   │   ├── train_vae_fsq.py           # Stage I: FSQ tokenizer (layers=4)
│   │   ├── train_vae_ddcl.py          # Stage I: DDCL tokenizer (layers=4)
│   │   └── train_maskgit.py           # Stage II: MaskGit transformer
│   └── cifar10/                       # CIFAR-10 128×128 training entry points
│       ├── train_vae_vq.py            # Stage I: VQ tokenizer (layers=3)
│       ├── train_vae_fsq.py           # Stage I: FSQ tokenizer (layers=3)
│       ├── train_vae_ddcl.py          # Stage I: DDCL tokenizer (layers=3)
│       └── train_maskgit.py           # Stage II: MaskGit transformer
└── image_generation/                  # Core library package
    ├── ddcl_quantizer.py              # DDCL quantizer implementation
    ├── vqgan_vae.py                   # VQGanVAE, FSQGanVAE, DDCLGanVAE + encoder/decoder/discriminator
    ├── muse_maskgit_pytorch.py        # MaskGit transformer, Muse, attention, sampling
    ├── trainers.py                    # VQGanVAETrainer (Stage I training loop, tqdm progress bars)
    ├── attend.py                      # Flash/memory-efficient attention
    └── t5.py                          # T5 text encoder for conditioning
```

---

## Key Abstractions

### Quantizers (the core comparison)

| Method | Class | Location | Codebook Size | Aux Loss |
|--------|-------|----------|---------------|----------|
| **VQ** | `VQGanVAE` | `vqgan_vae.py` L321 | 1024 | commitment + diversity |
| **FSQ** | `FSQGanVAE` | `vqgan_vae.py` L608 | 1024 (4⁵) | None |
| **DDCL** | `DDCLGanVAE` | `vqgan_vae.py` L659 | 1024 (4⁵) | λ·communication cost |

All three share identical encoder/decoder architecture (`dim=256, channels=3`) for fair comparison.
- **ImageNet**: `layers=4` (16× downsampling, 256→16×16 latent)
- **CIFAR-10**: `layers=3` (8× downsampling, 128→16×16 latent)

### Model Components

- **`DDCL`** (`ddcl_quantizer.py`) — Stochastic quantizer: project → tanh bound → uniform noise → floor → STE. Returns `(output, indices, comm_loss)`.
- **`ResnetEncDec`** (`vqgan_vae.py` L189) — Multi-layer encoder/decoder with `ResBlock` and `GLUResBlock`.
- **`Discriminator`** (`vqgan_vae.py` L155) — PatchGAN discriminator.
- **`MaskGitTransformer`** (`muse_maskgit_pytorch.py` L318) — Masked token prediction transformer with T5 cross-attention.
- **`MaskGit`** (`muse_maskgit_pytorch.py` L345) — Wraps frozen VAE + transformer. Iterative parallel decoding with cosine schedule.
- **`VQGanVAETrainer`** (`trainers.py` L117) — Stage I training loop with HuggingFace Accelerate, EMA, dual optimizers.
- **`Attend`** (`attend.py` L40) — Flash attention with fallback to memory-efficient attention.

---

## Training Pipeline

### Stage I — VAE Tokenizer

| Dataset | Entry points | Layers | Steps | Batch |
|---------|-------------|--------|-------|-------|
| ImageNet 256 | `scripts/imagenet/train_vae_{vq,fsq,ddcl}.py` | 4 | 1M | 512 |
| CIFAR-10 128 | `scripts/cifar10/train_vae_{vq,fsq,ddcl}.py` | 3 | 200k | 512 |

```bash
# Example: ImageNet DDCL
accelerate launch scripts/imagenet/train_vae_ddcl.py \
  --image_folder ./data/imagenet-256/train \
  --results_folder ./results/ddcl

# Example: CIFAR-10 DDCL
accelerate launch scripts/cifar10/train_vae_ddcl.py \
  --image_folder ./data/cifar10/train \
  --results_folder ./results/ddcl_cifar
```

- Loss = L1 reconstruction + VGG perceptual + hinge GAN + quantizer auxiliary + adaptive GAN weight
- Saves `vae.{step}.pt` and `vae.{step}.ema.pt` (EMA is the primary artifact)

### Stage II — MaskGit Transformer

| Dataset | Entry point | Steps | Batch |
|---------|------------|-------|-------|
| ImageNet 256 | `scripts/imagenet/train_maskgit.py` | 2.5M | 256 |
| CIFAR-10 128 | `scripts/cifar10/train_maskgit.py` | 500k | 256 |

Entry point: `scripts/imagenet/train_maskgit.py` or `scripts/cifar10/train_maskgit.py`

```bash
accelerate launch scripts/imagenet/train_maskgit.py \
  --vae_type ddcl \
  --vae_path ./results/ddcl/vae.1000000.ema.pt \
  --image_folder ./data/imagenet-256/train \
  --results_folder ./results/ddcl_maskgit
```

- Loads frozen Stage I VAE, trains MaskGitTransformer on masked token prediction
- Class labels → T5 text embeddings → cross-attention conditioning

---

## Environment Setup

```bash
cd image-generation
pip install -e .
accelerate config    # configure multi-GPU/distributed training

# Download ImageNet data
python scripts/download_imagenet.py --output_dir ./data/imagenet-256

# CIFAR-10 data: pre-resized 128×128 images in ImageFolder layout expected at e.g. ./data/cifar10/train/<class>/
```

**Requirements:** Python ≥ 3.12, PyTorch ≥ 2.9.1, CUDA-capable GPU(s).

Key dependencies: `accelerate`, `vector-quantize-pytorch`, `ema-pytorch`, `transformers`, `einops`, `beartype`, `memory-efficient-attention-pytorch`.

---

## Coding Conventions

1. **Utility helpers** — `exists(val)`, `default(val, d)`, `cycle(dl)` are defined locally in multiple modules as private utilities. Follow this pattern; do not extract into a shared module.
2. **Tensor manipulation** — Use `einops` (`rearrange`, `repeat`, `reduce`, `pack`/`unpack`) instead of raw PyTorch reshapes/permutes.
3. **Type checking** — Use `@beartype` decorator on public APIs for runtime validation. Use `typing` types (`Optional`, `List`, `Union`).
4. **Decorator patterns** — `@eval_decorator` for toggling eval/train mode, `@remove_vgg` for excluding VGG from state dicts.
5. **Kwargs routing** — `groupby_prefix_and_trim()` to dispatch prefixed kwargs (e.g., `vq_` prefix → VQ config).
6. **Config** — All entry points use `argparse` with defaults matching paper configurations. No YAML/JSON config files.
7. **Docstrings** — Module-level docstrings on entry-point scripts with usage examples. Class docstrings with paper references on key components.

---

## Package Name Note

The local package directory is `image_generation/` but train scripts import from `muse_maskgit_pytorch` (e.g., `from muse_maskgit_pytorch import DDCLGanVAE`). This package is a fork of the open-source `muse-maskgit-pytorch` with DDCL support added. The `pyproject.toml` declares `packages = ["image_generation"]` — the import name mapping is handled there.

---

## Data Loading

| Stage | Dataset | Loader | Location |
|-------|---------|--------|----------|
| Download | ImageNet-1k 256×256 | HuggingFace `datasets` | `scripts/download_imagenet.py` |
| Stage I | ImageNet / CIFAR-10 | `ImageDataset` (glob `**/*.{jpg,jpeg,png}`) | `trainers.py` L90 |
| Stage II | ImageNet / CIFAR-10 | `torchvision.ImageFolder` (class labels) | `scripts/*/train_maskgit.py` |

---

## Testing

There is currently **no test suite**. When adding tests, consider:
- Unit tests for quantizer forward/backward passes (verify STE gradients, codebook sizes)
- Integration tests for VAE encode → quantize → decode roundtrip
- Snapshot tests for deterministic generation with fixed seeds

---

## Checkpoint Format

- **Stage I**: `vae.{step}.pt` — raw `state_dict()` of the VAE. `.ema.pt` variant for EMA weights (primary artifact for Stage II).
- **Stage II**: `maskgit.{step}.pt` — `maskgit.state_dict()`, may be wrapped as `{"model": state_dict}`.
- Loading logic handles both wrapped and raw formats.
