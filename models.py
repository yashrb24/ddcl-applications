import torch.nn as nn
from quantizers import FSQWrapper, DDCL_Bottleneck, VanillaVAE, VQVAEWrapper, AEWrapper


class Encoder(nn.Module):
    """Encoder: Downsample image to latent representation"""

    def __init__(self, in_channels=3, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.Flatten(),
            nn.Linear(48 * 4 * 4, latent_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    """Decoder: Upsample latent to image"""

    def __init__(self, latent_dim=4, out_channels=3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 48 * 4 * 4),
            nn.Unflatten(dim=1, unflattened_size=(48, 4, 4)),
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, out_channels, 4, stride=2, padding=1),  # [batch, 3, 32, 32]
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)


class QuantizedVAE(nn.Module):
    """Generalized VAE with pluggable quantization methods"""

    def __init__(self, quantizer_type="fsq", levels=None, delta=None, codebook_size=None, latent_dim=4):
        super().__init__()

        self.quantizer_type = quantizer_type

        # Determine latent dimension
        encoder_latent_dim = None
        decoder_latent_dim = None

        match quantizer_type:
            case "fsq":
                if levels is None:
                    levels = [8, 5, 5, 5]
                latent_dim = len(levels)
                encoder_latent_dim = latent_dim
                decoder_latent_dim = latent_dim
            case "ddcl":
                encoder_latent_dim = latent_dim
                decoder_latent_dim = latent_dim
            case "vae":
                encoder_latent_dim = 2 * latent_dim
                decoder_latent_dim = latent_dim
            case "vq_vae":
                encoder_latent_dim = latent_dim
                decoder_latent_dim = latent_dim
            case "autoencoder":
                encoder_latent_dim = latent_dim
                decoder_latent_dim = latent_dim
            case _:
                raise ValueError(f"Unknown quantizer_type: {quantizer_type}")

        self.encoder = Encoder(latent_dim=encoder_latent_dim)
        self.decoder = Decoder(latent_dim=decoder_latent_dim)

        # Initialize quantizer
        if quantizer_type == "fsq":
            self.quantizer = FSQWrapper(levels=levels)
        elif quantizer_type == "ddcl":
            self.quantizer = DDCL_Bottleneck(delta=delta, latent_dim=latent_dim)
        elif quantizer_type == "vae":
            self.quantizer = VanillaVAE(latent_dim=latent_dim)
        elif quantizer_type == "vq_vae":
            self.quantizer = VQVAEWrapper(latent_dim=latent_dim, codebook_size=codebook_size)
        elif quantizer_type == "autoencoder":
            self.quantizer = AEWrapper()
        else:
            raise ValueError(f"Unknown quantizer_type: {quantizer_type}")

    def forward(self, x):
        """Forward pass through encoder -> quantizer -> decoder"""
        z = self.encoder(x)

        # Quantize (returns z_q, indices, reg_loss)
        z_q, indices, reg_loss = self.quantizer(z)

        recon = self.decoder(z_q)

        return recon, indices, reg_loss
