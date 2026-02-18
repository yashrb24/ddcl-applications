"""
Stage 1: Train FSQ-GAN VAE tokenizer on CIFAR-10 128×128.

FSQ levels: [4, 4, 4, 4, 4] → codebook size 1024
Downsampling: 8× (3 layers) → 16×16 latent map

Usage:
  accelerate launch scripts/cifar10/train_vae_fsq.py \
      --image_folder /path/to/cifar10/train \
      --results_folder ./results/vae_fsq_cifar
"""

import argparse
from muse_maskgit_pytorch import FSQGanVAE, VQGanVAETrainer


def main():
    parser = argparse.ArgumentParser(description="Train FSQ-GAN VAE tokenizer (CIFAR-10)")
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--results_folder", type=str, default="./results/vae_fsq_cifar")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--grad_accum_every", type=int, default=1)
    parser.add_argument("--num_train_steps", type=int, default=200_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_results_every", type=int, default=1000)
    parser.add_argument("--save_model_every", type=int, default=10000)
    args = parser.parse_args()

    vae = FSQGanVAE(
        dim=256,
        channels=3,
        layers=3,
        fsq_levels=[4, 4, 4, 4, 4],       # codebook_size = 4^5 = 1024
        l2_recon_loss=False,
        use_hinge_loss=True,
        use_vgg_and_gan=True,
        discr_layers=4,
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
