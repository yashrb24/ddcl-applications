from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloaders(dataset_name="CIFAR10", batch_size=16, num_workers=0, pin_memory=False):
    """
    Create train and validation dataloaders for the specified dataset.

    Args:
        dataset_name: Name of the dataset (currently only CIFAR10 supported)
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to use pinned memory for faster GPU transfer

    Returns:
        tuple: (train_loader, val_loader)
    """
    print("=" * 70)
    print(f"Loading {dataset_name} dataset...")

    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if dataset_name == "CIFAR10":
        # Load CIFAR10 dataset
        train_dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        val_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Currently only CIFAR10 is supported.")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    print(f"Batch size: {batch_size}")
    print("=" * 70)

    return train_loader, val_loader
