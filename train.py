import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from pathlib import Path

from models import QuantizedVAE
from utils import visualize_reconstructions, compute_codebook_usage, save_checkpoint


def train_epoch(model, dataloader, optimizer, device, comm_loss_weight=0.01):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_reg_loss = 0

    for batch_idx, (data, _) in enumerate(tqdm(dataloader, desc="Training")):
        data = data.to(device)
        optimizer.zero_grad()

        recon, _, reg_loss = model(data)

        recon_loss = F.mse_loss(recon, data)

        # Add regularization loss (weighted)
        total_batch_loss = recon_loss + comm_loss_weight * reg_loss

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


def main():
    # ======================== CONFIGURATION ========================
    QUANTIZER_TYPE = "fsq"  # 'fsq' or 'ddcl'

    # Training hyperparameters
    batch_size = 64
    epochs = 50
    lr = 3e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # FSQ settings
    fsq_levels = [8, 8, 8, 8]  # Codebook size = 8*8*8*8 = 4096

    # DDCL settings
    ddcl_delta = 1 / 10  # Quantization grid width
    ddcl_comm_weight = 1e-4  # Weight for communication loss

    # Paths
    output_dir = Path(f"outputs")
    checkpoint_dir = Path("checkpoints")
    output_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    # ======================== DATA LOADING ========================
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    val_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
    )

    # ======================== MODEL SETUP ========================
    if QUANTIZER_TYPE == "fsq":
        model = QuantizedVAE(quantizer_type="fsq", levels=fsq_levels).to(device)
        print("=" * 70)
        print("Training FSQ-VAE")
        print(f"Codebook size: {model.quantizer.codebook_size}")
        comm_loss_weight = 0.0  # No regularization loss for FSQ
    else:
        model = QuantizedVAE(quantizer_type="ddcl", delta=ddcl_delta).to(device)
        print("=" * 70)
        print("Training DDCL-VAE")
        print(f"Quantization Delta: {ddcl_delta}")
        print(f"Communication Loss Weight: {ddcl_comm_weight}")
        comm_loss_weight = ddcl_comm_weight

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 70)

    # ======================== TRAINING LOOP ========================
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, comm_loss_weight
        )

        # Validate
        val_loss = validate(model, val_loader, device)

        # Print metrics
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(
            f"  Train - Total: {train_metrics['total_loss']:.4f}, "
            f"Recon: {train_metrics['recon_loss']:.4f}, "
            f"Reg: {train_metrics['reg_loss']:.4f}"
        )
        print(f"  Val Recon Loss: {val_loss:.4f}")

        # Visualize reconstructions
        visualize_reconstructions(
            model,
            val_loader,
            device,
            epoch + 1,
            quantizer=QUANTIZER_TYPE,
            save_dir=output_dir,
        )

        # Compute codebook usage (FSQ only)
        if QUANTIZER_TYPE == "fsq" and (epoch + 1) % 5 == 0:
            stats = compute_codebook_usage(model, val_loader, device)
            if stats:
                print(
                    f"  Codebook usage: {stats['unique_codes']}/{stats['total_codes']} "
                    f"({stats['usage_percent']:.1f}%)"
                )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = checkpoint_dir / f"{QUANTIZER_TYPE}_vae_best.pt"
            save_checkpoint(model, optimizer, epoch + 1, val_loss, best_path)
            print(f"   New best validation loss: {val_loss:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = (
                checkpoint_dir / f"{QUANTIZER_TYPE}_vae_epoch_{epoch+1}.pt"
            )
            save_checkpoint(model, optimizer, epoch + 1, val_loss, checkpoint_path)

        print("-" * 70)

    print("\n" + "=" * 70)
    print(f"Training complete! Best validation loss: {best_val_loss:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
