"""
Stage 2: Train MaskGit transformer on ImageNet 256×256 with a frozen VAE tokenizer.

Works with any Stage 1 tokenizer (VQ / FSQ / DDCL).

The MaskGit transformer is class-conditional — ImageNet class names are used as
text prompts via T5.  Each image folder should be named after the class
(standard ImageFolder layout: imagenet/train/n01440764/...).

Usage:
  accelerate launch scripts/train_maskgit.py \
      --vae_type vq \
      --vae_path ./results/vae_vq/vae.1000000.ema.pt \
      --image_folder /path/to/imagenet/train \
      --results_folder ./results/maskgit_vq
"""

import argparse
import math
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

from accelerate import Accelerator

from tqdm.auto import tqdm

from muse_maskgit_pytorch import (
    VQGanVAE,
    FSQGanVAE,
    DDCLGanVAE,
    MaskGitTransformer,
    MaskGit,
)

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ── helpers ──────────────────────────────────────────────────────────────────

def cycle(dl):
    while True:
        for batch in dl:
            yield batch


def build_vae(vae_type: str) -> VQGanVAE:
    """Instantiate the correct VAE class (weights loaded separately)."""
    common = dict(dim=256, channels=3, layers=4, use_vgg_and_gan=False)

    if vae_type == "vq":
        return VQGanVAE(
            codebook_size=1024,
            lookup_free_quantization=False,
            **common,
        )
    elif vae_type == "fsq":
        return FSQGanVAE(
            fsq_levels=[4, 4, 4, 4, 4],
            **common,
        )
    elif vae_type == "ddcl":
        return DDCLGanVAE(
            n_dims=5,
            delta=1.0,
            scale=1.0,
            ddcl_lambda=1e-3,
            **common,
        )
    else:
        raise ValueError(f"Unknown vae_type: {vae_type}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train MaskGit (Stage 2)")
    parser.add_argument("--vae_type", type=str, required=True, choices=["vq", "fsq", "ddcl"])
    parser.add_argument("--vae_path", type=str, required=True, help="Path to Stage 1 VAE checkpoint")
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--results_folder", type=str, default="./results/maskgit")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--grad_accum_every", type=int, default=1)
    parser.add_argument("--num_train_steps", type=int, default=2_500_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_model_every", type=int, default=10000)
    parser.add_argument("--t5_name", type=str, default="t5-small")
    parser.add_argument("--transformer_depth", type=int, default=24)
    parser.add_argument("--transformer_dim", type=int, default=512)
    parser.add_argument("--dim_head", type=int, default=64)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--cond_drop_prob", type=float, default=0.25)
    parser.add_argument("--timesteps", type=int, default=12, help="Sampling steps (cosine schedule)")
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device

    # ── load frozen VAE ──────────────────────────────────────────────────

    vae = build_vae(args.vae_type)
    state_dict = torch.load(args.vae_path, map_location="cpu")
    # handle both raw state_dict and wrapped checkpoint formats
    if "model" in state_dict:
        state_dict = state_dict["model"]
    vae.load_state_dict(state_dict)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    codebook_size = vae.codebook_size
    fmap_size = vae.get_encoded_fmap_size(args.image_size)
    seq_len = fmap_size ** 2
    accelerator.print(f"VAE type={args.vae_type}, codebook_size={codebook_size}, "
                      f"fmap_size={fmap_size}, seq_len={seq_len}")

    # ── build MaskGit ────────────────────────────────────────────────────

    transformer = MaskGitTransformer(
        num_tokens=codebook_size,
        seq_len=seq_len,
        dim=args.transformer_dim,
        depth=args.transformer_depth,
        dim_head=args.dim_head,
        heads=args.heads,
        ff_mult=4,
        t5_name=args.t5_name,
    )

    maskgit = MaskGit(
        vae=vae,
        transformer=transformer,
        image_size=args.image_size,
        cond_drop_prob=args.cond_drop_prob,
    )

    # ── dataset ──────────────────────────────────────────────────────────

    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize(args.image_size),
        T.CenterCrop(args.image_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])

    dataset = ImageFolder(args.image_folder, transform=transform)
    # extract class name from folder name for text conditioning
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    accelerator.print(f"Dataset: {len(dataset)} images, {len(idx_to_class)} classes")

    dl = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # ── optimizer ────────────────────────────────────────────────────────

    optimizer = Adam(maskgit.transformer.parameters(), lr=args.lr)

    # ── prepare with accelerate ──────────────────────────────────────────

    maskgit, optimizer, dl = accelerator.prepare(maskgit, optimizer, dl)
    dl_iter = cycle(dl)

    results_folder = Path(args.results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)

    # ── training loop ────────────────────────────────────────────────────

    pbar = tqdm(
        range(1, args.num_train_steps + 1),
        desc="maskgit training",
        disable=not accelerator.is_main_process,
    )

    for step in pbar:
        maskgit.train()

        total_loss = 0.0
        for _ in range(args.grad_accum_every):
            images, class_indices = next(dl_iter)
            images = images.to(device)

            # use class folder names as text prompts
            texts = [idx_to_class[idx.item()] for idx in class_indices]

            with accelerator.autocast():
                loss = maskgit(images, texts=texts)

            accelerator.backward(loss / args.grad_accum_every)
            total_loss += loss.item() / args.grad_accum_every

        optimizer.step()
        optimizer.zero_grad()

        pbar.set_postfix(loss=f"{total_loss:.4f}")

        if step % 100 == 0:
            tqdm.write(f"step {step}: loss = {total_loss:.4f}")

        # ── save checkpoint ──────────────────────────────────────────────

        if step % args.save_model_every == 0 and accelerator.is_main_process:
            ckpt_path = results_folder / f"maskgit.{step}.pt"
            accelerator.save(
                accelerator.unwrap_model(maskgit).state_dict(),
                str(ckpt_path),
            )
            accelerator.print(f"step {step}: saved checkpoint to {ckpt_path}")

    accelerator.print("Training complete.")


if __name__ == "__main__":
    main()
