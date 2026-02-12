"""
Stage 1: Train VQ-GAN VAE tokenizer on ImageNet 256×256.

Codebook size: 1024 (standard VQ)
Downsampling: 16× (4 layers) → 16×16 latent map

Usage:
  accelerate launch scripts/train_vae_vq.py \
      --image_folder /path/to/imagenet/train \
      --results_folder ./results/vae_vq
"""

import argparse
from muse_maskgit_pytorch import VQGanVAE, VQGanVAETrainer


def main():
    parser = argparse.ArgumentParser(description="Train VQ-GAN VAE tokenizer")
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--results_folder", type=str, default="./results/vae_vq")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--grad_accum_every", type=int, default=1)
    parser.add_argument("--num_train_steps", type=int, default=1_000_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_results_every", type=int, default=1000)
    parser.add_argument("--save_model_every", type=int, default=10000)
    args = parser.parse_args()

    vae = VQGanVAE(
        dim=256,
        channels=3,
        layers=4,
        codebook_size=1024,
        lookup_free_quantization=False,
        l2_recon_loss=False,
        use_hinge_loss=True,
        use_vgg_and_gan=True,
        discr_layers=4,
        vq_kwargs=dict(
            codebook_dim=256,
            decay=0.99,
            commitment_weight=1.0,
            kmeans_init=True,
            use_cosine_sim=True,
            codebook_diversity_loss_weight=1.0,  # entropy loss for uniform codebook usage (paper §4.1)
            codebook_diversity_temperature=100.,
        ),
    )

    trainer = VQGanVAETrainer(
        vae=vae,
        folder=args.image_folder,
        image_size=args.image_size,
        batch_size=args.batch_size,
        grad_accum_every=args.grad_accum_every,
        num_train_steps=args.num_train_steps,
        lr=args.lr,
        save_results_every=args.save_results_every,
        save_model_every=args.save_model_every,
        results_folder=args.results_folder,
    )

    trainer.train()


if __name__ == "__main__":
    main()
