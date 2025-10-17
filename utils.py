from pathlib import Path

import matplotlib.pyplot as plt
import torch

import wandb


@torch.no_grad()
def visualize_reconstructions_new_arch(
        model, dataloader, device, epoch, quantizer, save_dir="outputs", use_wandb=False, run_name=None
):
    """
    Generate and save a 4x4 grid of original and reconstructed images

    Args:
        model: Quantized VAE model
        dataloader: DataLoader to sample images from
        device: Device to run model on
        epoch: Current epoch number
        quantizer: Type of quantizer ('fsq' or 'ddcl')
        save_dir: Directory to save images
        use_wandb: Whether to log images to wandb
        run_name: Run-specific name for organizing local files (e.g., 'delta0.1_weight1e-4')
    """
    model.eval()

    # Create directory structure
    if run_name:
        # Parameter-specific folder for sweep runs
        output_path = Path(save_dir) / run_name
    else:
        # Legacy behavior: organize by quantizer type only
        output_path = Path(save_dir) / quantizer

    output_path.mkdir(parents=True, exist_ok=True)

    # Get a batch of images
    images, _ = next(iter(dataloader))
    images = images[:8].to(device)  # Take 8 images for 4x4 grid

    # Get reconstructions
    reconstructions, _, _ = model(images)

    # Create figure with original and reconstructed images
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle(
        f"Epoch {epoch} - Top 2 rows: Original, Bottom 2 rows: Reconstructed",
        fontsize=14,
        fontweight="bold",
    )

    # Plot images in 4x4 grid
    for idx in range(8):
        row = idx // 4
        col = idx % 4

        # Original images (top 2 rows)
        img = images[idx].permute(1, 2, 0).cpu().numpy()
        axes[row, col].imshow(img)
        axes[row, col].axis("off")

        # Reconstructed images (bottom 2 rows)
        recon = reconstructions[idx].permute(1, 2, 0).cpu().numpy()
        axes[row + 2, col].imshow(recon)
        axes[row + 2, col].axis("off")

    plt.tight_layout()

    # Save figure locally
    save_path = output_path / f"reconstruction_epoch_{epoch:03d}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    # Log to wandb if enabled (log every 5 epochs to reduce bandwidth)
    if use_wandb and epoch % 10 == 0:
        wandb.log({"reconstructions": wandb.Image(fig)}, step=epoch)

    plt.close()

    print(f"  → Saved reconstruction grid to {save_path}")


@torch.no_grad()
def compute_codebook_usage(model, dataloader, device, num_batches=10):
    """
    Compute which codebook indices are being used (only for FSQ)

    Args:
        model: Quantized VAE model
        dataloader: DataLoader
        device: Device
        num_batches: Number of batches to analyze

    Returns:
        Dictionary with usage statistics or None if not applicable
    """
    if model.quantizer_type != "fsq":
        return None

    model.eval()
    all_indices = []

    for batch_idx, (images, _) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        images = images.to(device)
        _, indices, _ = model(images)

        if indices is not None:
            all_indices.append(indices.cpu())

    if not all_indices:
        return None

    all_indices = torch.cat(all_indices, dim=0)
    unique_indices = torch.unique(all_indices)

    # Calculate total possible codes
    total_codes = model.quantizer.codebook_size
    usage_percent = (len(unique_indices) / total_codes) * 100

    stats = {
        "unique_codes": len(unique_indices),
        "total_codes": total_codes,
        "usage_percent": usage_percent,
    }

    return stats


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        filepath,
    )
    print(f"  → Saved checkpoint to {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["loss"]
