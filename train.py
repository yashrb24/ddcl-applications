import argparse
from pathlib import Path

import torch
import torch.nn as nn
import wandb

from dataloading import get_dataloaders
from models import create_model
from train_utils import train_epoch, validate
from utils import compute_codebook_usage, save_checkpoint, visualize_reconstructions_new_arch

# Perceptual loss computation interval (every N epochs)
PERCEPTUAL_LOSS_INTERVAL = 10


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Quantized VAE")

    # Model configuration
    parser.add_argument("--quantizer_type", type=str, default="fsq",
                        choices=["fsq", "ddcl", "vae", "vq_vae", "autoencoder"],
                        help="Quantizer type: 'fsq' or 'ddcl' or 'vae' or 'vq_vae' or 'autoencoder'")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    # FSQ settings
    parser.add_argument("--fsq_levels", type=int, nargs="+", default=[8, 8, 8, 8],
                        help="FSQ levels (codebook size = product of levels)")

    # DDCL settings
    parser.add_argument("--ddcl_delta", type=float, default=0.1, help="DDCL quantization grid width")
    parser.add_argument("--reg_loss_weight", type=float, default=1e-4,
                        help="regularization loss weight, KL loss weight for VAE, commitment loss for vqvae, communication loss weight for DDCL")

    # VQ-VAE settings
    parser.add_argument("--codebook_size", type=int, default=128, help="VQ-VAE codebook size")

    # General quantizer settings
    parser.add_argument("--latent_dim", type=int, default=4,
                        help="Latent space dimensionality (used for all quantizers except FSQ)")

    # Wandb settings
    parser.add_argument("--use_wandb", type=lambda x: x.lower() == 'true',
                        default=False, help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="ddcl-vae", help="Wandb project name")

    return parser.parse_args()


def main():
    args = parse_args()

    # ======================== CONFIGURATION ========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb if requested
    config = None
    if args.use_wandb:
        # Check if wandb run already exists (from sweep agent)
        if wandb.run is None:
            wandb.init(project=args.wandb_project, config=vars(args))
            config = wandb.config
    else:
        config = args

    # Create a run-specific name for organizing outputs during sweeps
    run_name = None
    match config.quantizer_type:
        case "fsq":
            run_name = f"fsq_levels{config.fsq_levels}"
        case "ddcl":
            run_name = f"ddcl_delta{config.ddcl_delta}_weight{config.reg_loss_weight}"
        case "vae":
            run_name = f"vae"
        case "vq_vae":
            run_name = f"vq_vae"
        case "autoencoder":
            run_name = f"autoencoder"
        case _:
            raise ValueError(f"Unknown quantizer_type: {config.quantizer_type}")

    # Paths
    output_dir = Path("outputs")
    checkpoint_dir = Path("checkpoints")
    output_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    # ======================== DATA LOADING ========================
    train_loader, val_loader = get_dataloaders(
        dataset_name="CIFAR10",
        batch_size=config.batch_size,
        num_workers=0,
        pin_memory=False
    )

    # ======================== MODEL SETUP ========================
    model = create_model(
        quantizer_type=config.quantizer_type,
        device=device,
        fsq_levels=config.fsq_levels,
        ddcl_delta=config.ddcl_delta,
        codebook_size=config.codebook_size,
        latent_dim=config.latent_dim
    )

    # Configure regularization loss weight based on quantizer type
    reg_loss_weight = None
    match config.quantizer_type:
        case "fsq":
            reg_loss_weight = 0.0  # No regularization loss for FSQ
        case "vae":
            reg_loss_weight = config.reg_loss_weight
            print(f"KL Loss Weight: {config.reg_loss_weight}")
        case "vq_vae":
            reg_loss_weight = config.reg_loss_weight
            print(f"Commitment Loss Weight: {config.reg_loss_weight}")
        case "ddcl":
            reg_loss_weight = config.reg_loss_weight
            print(f"Communication Loss Weight: {config.reg_loss_weight}")
        case "autoencoder":
            reg_loss_weight = 0.0  # No regularization loss for autoencoder

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.BCELoss()

    # ======================== TRAINING LOOP ========================
    best_val_loss = float("inf")

    for epoch in range(config.epochs):
        # Determine if we should compute perceptual loss this epoch
        should_compute_perceptual = (epoch + 1) % PERCEPTUAL_LOSS_INTERVAL == 0

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, reg_loss_weight,
            compute_perceptual=should_compute_perceptual
        )

        # Validate
        val_recon_loss, val_perceptual_loss = validate(
            model, val_loader, criterion, device, compute_perceptual=should_compute_perceptual
        )

        # Log metrics to wandb
        if args.use_wandb:
            log_dict = {
                "epoch": epoch + 1,
                "train/total_loss": train_metrics['total_loss'],
                "train/recon_loss": train_metrics['recon_loss'],
                "train/reg_loss": train_metrics['reg_loss'],
                "val/recon_loss": val_recon_loss,
            }
            # Only log perceptual loss when computed
            if should_compute_perceptual:
                log_dict["train/perceptual_loss"] = train_metrics['perceptual_loss']
                log_dict["val/perceptual_loss"] = val_perceptual_loss
            wandb.log(log_dict)

        # Print metrics
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        train_msg = (
            f"  Train - Total: {train_metrics['total_loss']:.4f}, "
            f"Recon: {train_metrics['recon_loss']:.4f}, "
            f"Reg: {train_metrics['reg_loss']:.4f}"
        )
        if should_compute_perceptual:
            train_msg += f", Perceptual: {train_metrics['perceptual_loss']:.4f}"
        print(train_msg)

        val_msg = f"  Val - Recon Loss: {val_recon_loss:.4f}"
        if should_compute_perceptual:
            val_msg += f", Perceptual: {val_perceptual_loss:.4f}"
        print(val_msg)

        # Visualize reconstructions
        visualize_reconstructions_new_arch(
            model,
            val_loader,
            device,
            epoch + 1,
            quantizer=config.quantizer_type,
            save_dir=output_dir,
            use_wandb=args.use_wandb,
            run_name=run_name,
        )

        # Compute codebook usage (FSQ, VQ-VAE, and DDCL)
        if config.quantizer_type in ["fsq", "vq_vae", "ddcl"]:
            codebook_metrics = compute_codebook_usage(model, val_loader, device)
            # Log codebook metrics to wandb
            if codebook_metrics and args.use_wandb:
                wandb.log(codebook_metrics)

        # Save best model
        if val_recon_loss < best_val_loss:
            best_val_loss = val_recon_loss
            best_path = checkpoint_dir / f"{config.quantizer_type}_vae_best.pt"
            save_checkpoint(model, optimizer, epoch + 1, val_recon_loss, best_path)
            print(f"   New best validation loss: {val_recon_loss:.4f}")

        # Log best model to wandb
        # if args.use_wandb:
        #     wandb.log({"best_val_loss": best_val_loss})

        # Save a checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = (
                    checkpoint_dir / f"{config.quantizer_type}_vae_epoch_{epoch + 1}.pt"
            )
            save_checkpoint(model, optimizer, epoch + 1, val_recon_loss, checkpoint_path)

        print("-" * 70)

    print("\n" + "=" * 70)
    print(f"Training complete! Best validation loss: {best_val_loss:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
