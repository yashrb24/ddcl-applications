import torch.nn as nn
from quantizers import FSQWrapper, DDCL_Bottleneck


class Encoder(nn.Module):
    """Encoder: Downsample image to latent representation"""

    def __init__(self, in_channels=3, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),  # 32->16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16->8
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 8->4
            nn.ReLU(),
            nn.Conv2d(256, latent_dim, kernel_size=1),  # Project to latent_dim
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    """Decoder: Upsample latent to image"""

    def __init__(self, latent_dim=4, out_channels=3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, 256, kernel_size=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 4->8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 8->16
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, out_channels, kernel_size=4, stride=2, padding=1
            ),  # 16->32
        )

    def forward(self, x):
        return self.decoder(x)


class QuantizedVAE(nn.Module):
    """Generalized VAE with pluggable quantization methods"""

    def __init__(self, quantizer_type="fsq", levels=[8, 5, 5, 5], delta=0.1):
        super().__init__()
        self.quantizer_type = quantizer_type

        # Determine latent dimension
        if quantizer_type == "fsq":
            latent_dim = len(levels)
        elif quantizer_type == "ddcl":
            latent_dim = 4  # Fixed for DDCL, can be made configurable
        else:
            raise ValueError(f"Unknown quantizer_type: {quantizer_type}")

        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)

        # Initialize quantizer
        if quantizer_type == "fsq":
            self.quantizer = FSQWrapper(levels=levels)
        elif quantizer_type == "ddcl":
            self.quantizer = DDCL_Bottleneck(delta=delta, latent_dim=latent_dim)
        else:
            raise ValueError(f"Unknown quantizer_type: {quantizer_type}")

    def forward(self, x):
        """Forward pass through encoder -> quantizer -> decoder"""
        z = self.encoder(x)
        z = z.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)

        # Quantize (returns z_q, indices, reg_loss)
        z_q, indices, reg_loss = self.quantizer(z)

        z_q = z_q.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        recon = self.decoder(z_q)

        return recon, indices, reg_loss

    def encode(self, x):
        """Encode image to discrete indices"""
        z = self.encoder(x).permute(0, 2, 3, 1)
        if self.quantizer_type == "fsq":
            _, indices, _ = self.quantizer(z)
        else:  # ddcl
            _, indices = self.quantizer.quantize_and_dequantize(z)
        return indices
