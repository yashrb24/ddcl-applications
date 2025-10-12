import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from vector_quantize_pytorch import FSQ

from utils import visualize_reconstructions, compute_codebook_usage


# Encoder: Downsample image to latent representation
class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),  # 32->16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16->8
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 8->4
            nn.ReLU(),
            nn.Conv2d(256, latent_dim, kernel_size=1),  # 4->4, project to latent_dim
        )

    def forward(self, x):
        return self.encoder(x)


# Decoder: Upsample latent to image
class Decoder(nn.Module):
    def __init__(self, latent_dim=4, out_channels=3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, 256, kernel_size=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 4->8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 8->16
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),  # 16->32
        )

    def forward(self, x):
        return self.decoder(x)


# FSQ-VAE Model
class FSQVAE(nn.Module):
    def __init__(self, levels=[8, 5, 5, 5]):
        super().__init__()
        latent_dim = len(levels)
        self.encoder = Encoder(latent_dim=latent_dim)
        self.fsq = FSQ(levels=levels)
        self.decoder = Decoder(latent_dim=latent_dim)

    def forward(self, x):
        # Encode
        z = self.encoder(x)

        # Rearrange for FSQ: (B, C, H, W) -> (B, H, W, C)
        z = z.permute(0, 2, 3, 1)

        # Quantize with FSQ (no commitment loss needed!)
        z_q, indices = self.fsq(z)

        # Rearrange back: (B, H, W, C) -> (B, C, H, W)
        z_q = z_q.permute(0, 3, 1, 2)

        # Decode
        recon = self.decoder(z_q)

        return recon, indices

    def encode(self, x):
        z = self.encoder(x)
        z = z.permute(0, 2, 3, 1)
        _, indices = self.fsq(z)
        return indices

    def decode_from_indices(self, indices):
        z_q = self.fsq.indices_to_codes(indices)
        z_q = z_q.permute(0, 3, 1, 2)
        return self.decoder(z_q)


# Training function
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch_idx, (data, _) in enumerate(tqdm(dataloader, desc="Training")):
        data = data.to(device)

        optimizer.zero_grad()
        recon, _ = model(data)

        # Only reconstruction loss - FSQ doesn't need commitment loss!
        loss = F.mse_loss(recon, data)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# Validation function
@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0

    for data, _ in dataloader:
        data = data.to(device)
        recon, _ = model(data)
        loss = F.mse_loss(recon, data)
        total_loss += loss.item()

    return total_loss / len(dataloader)


# Main training script
def main():
    # Hyperparameters
    batch_size = 128
    epochs = 50
    lr = 3e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # FSQ levels - total codebook size = 8*5*5*5 = 1000
    levels = [8, 5, 5, 5]

    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # CIFAR-10 datasets
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    val_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Model, optimizer
    model = FSQVAE(levels=levels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"Training FSQ-VAE on {device}")
    print(f"Codebook size: {torch.prod(torch.tensor(levels)).item()}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Visualize reconstructions
        visualize_reconstructions(model, val_loader, device, epoch + 1, save_dir='outputs')

        # Compute codebook usage stats every 5 epochs
        if (epoch + 1) % 5 == 0:
            stats = compute_codebook_usage(model, val_loader, device)
            print(f"Codebook usage: {stats['unique_codes']}/{stats['total_codes']} "
                  f"({stats['usage_percent']:.1f}%)")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'fsq_vae_best.pt')
            print(f"Saved best model with val loss: {val_loss:.4f}")

        print("-" * 70)

    print("Training complete!")


if __name__ == "__main__":
    main()
