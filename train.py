import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import wandb
from models import QuantizedVAE
from utils import compute_codebook_usage, save_checkpoint, visualize_reconstructions_new_arch


def train_epoch(model, dataloader, optimizer, criterion, device, reg_loss_weight=0.01):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_reg_loss = 0

    for batch_idx, (data, _) in enumerate(tqdm(dataloader, desc="Training")):
        data = data.to(device)
        optimizer.zero_grad()

        recon, _, reg_loss = model(data)

        recon_loss = criterion(recon, data)

        # Add regularization loss (weighted)
        total_batch_loss = recon_loss + reg_loss_weight * reg_loss

        total_batch_loss.backward()
        optimizer.step()

        total_loss += total_batch_loss.item()
        total_recon_loss += recon_loss.item()
        total_reg_loss += reg_loss if isinstance(reg_loss, float) else reg_loss.item()

    return {
        "total_loss": total_loss / len(dataloader),
        "recon_loss": total_recon_loss / len(dataloader),
        "reg_loss": total_reg_loss / len(dataloader),
    }


@torch.no_grad()
def validate(model, dataloader, device):
    """Validate model"""
    model.eval()
    total_loss = 0

    for data, _ in dataloader:
        data = data.to(device)
        recon, _, _ = model(data)
        loss = F.mse_loss(recon, data)
        total_loss += loss.item()

    return total_loss / len(dataloader)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Quantized VAE")

    # Model configuration
    parser.add_argument("--quantizer_type", type=str, default="fsq", choices=["fsq", "ddcl", "vae"],
                        help="Quantizer type: 'fsq' or 'ddcl' or 'vae' ")

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
                        help="regularization loss weight, KL loss weight for VAE and communication loss weight for DDCL")

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

    # Create run-specific name for organizing outputs during sweeps
    if config.quantizer_type == "ddcl":
        run_name = f"ddcl_delta{config.ddcl_delta}_weight{config.reg_loss_weight}"
    elif config.quantizer_type == "vae":
        run_name = f"vae"
    else:
        run_name = f"fsq"

    # Paths
    output_dir = Path("outputs")
    checkpoint_dir = Path("checkpoints")
    output_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    # ======================== DATA LOADING ========================
    transform = transforms.Compose(
        [transforms.ToTensor()]
    )

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    val_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # ======================== MODEL SETUP ========================
    if config.quantizer_type == "fsq":
        model = QuantizedVAE(quantizer_type="fsq", levels=config.fsq_levels).to(device)
        print("=" * 70)
        print("Training FSQ-VAE")
        print(f"Codebook size: {model.quantizer.codebook_size}")
        reg_loss_weight = 0.0  # No regularization loss for FSQ
    elif config.quantizer_type == "vae":
        model = QuantizedVAE(quantizer_type="vae").to(device)
        print("=" * 70)
        print("Training Vanilla VAE")
        reg_loss_weight = config.reg_loss_weight
    else:
        model = QuantizedVAE(quantizer_type="ddcl", delta=config.ddcl_delta).to(device)
        print("=" * 70)
        print("Training DDCL-VAE")
        print(f"Quantization Delta: {config.ddcl_delta}")
        print(f"Communication Loss Weight: {config.reg_loss_weight}")
        reg_loss_weight = config.reg_loss_weight

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.BCELoss()
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 70)

    # ======================== TRAINING LOOP ========================
    best_val_loss = float("inf")

    for epoch in range(config.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, reg_loss_weight
        )

        # Validate
        val_loss = validate(model, val_loader, device)

        # Log metrics to wandb
        if args.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/total_loss": train_metrics['total_loss'],
                "train/recon_loss": train_metrics['recon_loss'],
                "train/reg_loss": train_metrics['reg_loss'],
                "val/recon_loss": val_loss,
            })

        # Print metrics
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        print(
            f"  Train - Total: {train_metrics['total_loss']:.4f}, "
            f"Recon: {train_metrics['recon_loss']:.4f}, "
            f"Reg: {train_metrics['reg_loss']:.4f}"
        )
        print(f"  Val Recon Loss: {val_loss:.4f}")

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

        # Compute codebook usage (FSQ only)
        if config.quantizer_type == "fsq" and (epoch + 1) % 5 == 0:
            stats = compute_codebook_usage(model, val_loader, device)
            if stats:
                print(
                    f"  Codebook usage: {stats['unique_codes']}/{stats['total_codes']} "
                    f"({stats['usage_percent']:.1f}%)"
                )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = checkpoint_dir / f"{config.quantizer_type}_vae_best.pt"
            save_checkpoint(model, optimizer, epoch + 1, val_loss, best_path)
            print(f"   New best validation loss: {val_loss:.4f}")

            # Log best model to wandb
            if args.use_wandb:
                wandb.log({"best_val_loss": best_val_loss})

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = (
                    checkpoint_dir / f"{config.quantizer_type}_vae_epoch_{epoch + 1}.pt"
            )
            save_checkpoint(model, optimizer, epoch + 1, val_loss, checkpoint_path)

        print("-" * 70)

    print("\n" + "=" * 70)
    print(f"Training complete! Best validation loss: {best_val_loss:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
