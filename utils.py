import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from pathlib import Path


def denormalize(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """Denormalize tensor from [-1, 1] to [0, 1]"""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


@torch.no_grad()
def visualize_reconstructions(model, dataloader, device, epoch, save_dir='outputs'):
    """
    Generate and save a 4x4 grid of original and reconstructed images

    Args:
        model: FSQ-VAE model
        dataloader: DataLoader to sample images from
        device: Device to run model on
        epoch: Current epoch number
        save_dir: Directory to save images
    """
    model.eval()

    # Create output directory
    Path(save_dir).mkdir(exist_ok=True)

    # Get a batch of images
    images, _ = next(iter(dataloader))
    images = images[:8].to(device)  # Take 8 images for 4x4 grid (4 original + 4 reconstructed)

    # Get reconstructions
    reconstructions, indices = model(images)

    # Denormalize for visualization
    images = denormalize(images.cpu())
    reconstructions = denormalize(reconstructions.cpu())

    # Clamp to [0, 1]
    images = torch.clamp(images, 0, 1)
    reconstructions = torch.clamp(reconstructions, 0, 1)

    # Create figure with original and reconstructed images
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle(f'Epoch {epoch} - Top 2 rows: Original, Bottom 2 rows: Reconstructed',
                 fontsize=14, fontweight='bold')

    # Plot images in 4x4 grid
    for idx in range(8):
        row = idx // 4
        col = idx % 4

        # Original images (top 2 rows)
        img = images[idx].permute(1, 2, 0).numpy()
        axes[row, col].imshow(img)
        axes[row, col].axis('off')

        # Reconstructed images (bottom 2 rows)
        recon = reconstructions[idx].permute(1, 2, 0).numpy()
        axes[row + 2, col].imshow(recon)
        axes[row + 2, col].axis('off')

    plt.tight_layout()

    # Save figure
    save_path = Path(save_dir) / f'reconstruction_epoch_{epoch:03d}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved reconstruction grid to {save_path}")


@torch.no_grad()
def visualize_grid(model, dataloader, device, epoch, save_dir='outputs', num_images=16):
    """
    Generate and save a grid of reconstructed images only

    Args:
        model: FSQ-VAE model
        dataloader: DataLoader to sample images from
        device: Device to run model on
        epoch: Current epoch number
        save_dir: Directory to save images
        num_images: Number of images to show
    """
    model.eval()

    # Create output directory
    Path(save_dir).mkdir(exist_ok=True)

    # Get images
    images, _ = next(iter(dataloader))
    images = images[:num_images].to(device)

    # Get reconstructions
    reconstructions, _ = model(images)

    # Denormalize
    images = denormalize(images.cpu())
    reconstructions = denormalize(reconstructions.cpu())

    # Clamp to [0, 1]
    images = torch.clamp(images, 0, 1)
    reconstructions = torch.clamp(reconstructions, 0, 1)

    # Concatenate original and reconstructed
    comparison = torch.cat([images, reconstructions])

    # Make grid
    grid = vutils.make_grid(comparison, nrow=4, padding=2, normalize=False)

    # Convert to numpy for matplotlib
    grid_np = grid.permute(1, 2, 0).numpy()

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(grid_np)
    ax.axis('off')
    ax.set_title(f'Epoch {epoch} - Top: Original, Bottom: Reconstructed',
                 fontsize=14, fontweight='bold', pad=10)

    # Save
    save_path = Path(save_dir) / f'grid_epoch_{epoch:03d}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved grid to {save_path}")


@torch.no_grad()
def compute_codebook_usage(model, dataloader, device, num_batches=10):
    """
    Compute which codebook indices are being used

    Args:
        model: FSQ-VAE model
        dataloader: DataLoader
        device: Device
        num_batches: Number of batches to analyze

    Returns:
        Dictionary with usage statistics
    """
    model.eval()

    all_indices = []

    for batch_idx, (images, _) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        images = images.to(device)
        _, indices = model(images)
        all_indices.append(indices.cpu())

    all_indices = torch.cat(all_indices, dim=0)
    unique_indices = torch.unique(all_indices)

    # Calculate total possible codes
    total_codes = model.fsq.codebook_size
    usage_percent = (len(unique_indices) / total_codes) * 100

    stats = {
        'unique_codes': len(unique_indices),
        'total_codes': total_codes,
        'usage_percent': usage_percent,
    }

    return stats


def save_sample_images(images, filename='sample.png'):
    """Save a tensor of images as a grid"""
    images = denormalize(images.cpu())
    images = torch.clamp(images, 0, 1)
    grid = vutils.make_grid(images, nrow=4, padding=2)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis('off')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()