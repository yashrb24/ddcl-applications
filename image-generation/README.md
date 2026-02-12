# Training Scripts

## Prerequisites

```bash
pip install muse-maskgit-pytorch accelerate datasets pillow
```

## 0. Download ImageNet 256×256

```bash
python scripts/download_imagenet.py --output_dir ./data/imagenet-256
```

This downloads from [evanarlian/imagenet_1k_resized_256](https://huggingface.co/datasets/evanarlian/imagenet_1k_resized_256) (~25 GB) and saves as ImageFolder layout with human-readable class names.

## 1. Stage I — Train VAE Tokenizer (1M steps, batch 512)

Three tokenizer variants, all producing codebook size 1024 with 16×16 latent maps:

```bash
# VQ-VAE (standard vector quantization)
accelerate launch scripts/train_vae_vq.py \
    --image_folder ./data/imagenet-256/train \
    --results_folder ./results/vae_vq

# FSQ-VAE (finite scalar quantization, levels=[4,4,4,4,4])
accelerate launch scripts/train_vae_fsq.py \
    --image_folder ./data/imagenet-256/train \
    --results_folder ./results/vae_fsq

# DDCL-VAE (differentiable discrete communication learning)
accelerate launch scripts/train_vae_ddcl.py \
    --image_folder ./data/imagenet-256/train \
    --results_folder ./results/vae_ddcl
```

Common flags:
- `--batch_size 512` (default, reduce if OOM)
- `--grad_accum_every 8` (increase to simulate larger batch on fewer GPUs)
- `--num_train_steps 1000000`
- `--lr 1e-4`
- `--save_model_every 10000`

## 2. Stage II — Train MaskGit Transformer (2.5M steps, batch 256)

Uses a frozen Stage I tokenizer. Works with any of the three VAE types:

```bash
# With VQ tokenizer
accelerate launch scripts/train_maskgit.py \
    --vae_type vq \
    --vae_path ./results/vae_vq/vae.1000000.ema.pt \
    --image_folder ./data/imagenet-256/train \
    --results_folder ./results/maskgit_vq

# With FSQ tokenizer
accelerate launch scripts/train_maskgit.py \
    --vae_type fsq \
    --vae_path ./results/vae_fsq/vae.1000000.ema.pt \
    --image_folder ./data/imagenet-256/train \
    --results_folder ./results/maskgit_fsq

# With DDCL tokenizer
accelerate launch scripts/train_maskgit.py \
    --vae_type ddcl \
    --vae_path ./results/vae_ddcl/vae.1000000.ema.pt \
    --image_folder ./data/imagenet-256/train \
    --results_folder ./results/maskgit_ddcl
```

Common flags:
- `--batch_size 256` (default)
- `--num_train_steps 2500000`
- `--transformer_depth 24`
- `--transformer_dim 512`
- `--t5_name t5-small`
- `--timesteps 12` (cosine schedule sampling steps)

## Tokenizer Comparison

| Tokenizer | Codebook Size | Quantized Dims | Aux Loss |
|-----------|--------------|----------------|----------|
| VQ        | 1024         | — (nearest neighbor) | commitment + codebook |
| FSQ       | 1024 (4⁵)   | 5              | none |
| DDCL      | 1024 (4⁵)   | 5              | communication cost |

## Multi-GPU

Configure accelerate first:

```bash
accelerate config
```

Then all `accelerate launch` commands above automatically use multi-GPU.
