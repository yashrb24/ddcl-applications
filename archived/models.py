import torch.nn as nn
from quantizers import FSQWrapper, DDCL_Bottleneck, VanillaVAE, VQVAEWrapper, AEWrapper

class Encoder(nn.Module):
    """Encoder: Less aggressive spatial downsampling, more channels → fewer channels"""
    def __init__(self, in_channels=3, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            # 32×32 → 16×16
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            
            # Stay at 16×16, increase channels
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            
            # Stay at 16×16, reduce to latent channels
            nn.Conv2d(64, latent_dim, 1),
        )
        # Output: [batch, latent_dim, 16, 16]
    
    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    """Decoder: Match encoder's spatial resolution"""
    def __init__(self, latent_dim=4, out_channels=3):
        super().__init__()
        self.decoder = nn.Sequential(
            # Start at 16×16, expand channels
            nn.Conv2d(latent_dim, 64, 1),
            nn.ReLU(),
            
            # Stay at 16×16
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            
            # 16×16 → 32×32
            nn.ConvTranspose2d(32, out_channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
        # Output: [batch, 3, 32, 32]
    
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


def create_model(quantizer_type, device, fsq_levels=None, ddcl_delta=None,
                 codebook_size=None, latent_dim=4):
    """
    Create a QuantizedVAE model with the specified quantizer type.

    Args:
        quantizer_type: Type of quantizer ('fsq', 'ddcl', 'vae', 'vq_vae', 'autoencoder')
        device: Device to place the model on
        fsq_levels: FSQ levels (list of ints) for FSQ quantizer
        ddcl_delta: Delta value for DDCL quantizer
        codebook_size: Codebook size for VQ-VAE quantizer
        latent_dim: Latent dimension for non-FSQ quantizers

    Returns:
        model: QuantizedVAE model instance on the specified device
    """
    print("=" * 70)

    model = None
    match quantizer_type:
        case "fsq":
            if fsq_levels is None:
                fsq_levels = [8, 8, 8, 8]
            model = QuantizedVAE(quantizer_type="fsq", levels=fsq_levels).to(device)
            print("Training FSQ-VAE")
            print(f"Codebook size: {model.quantizer.codebook_size}")

        case "vae":
            model = QuantizedVAE(quantizer_type="vae", latent_dim=latent_dim).to(device)
            print("Training Vanilla VAE")

        case "vq_vae":
            model = QuantizedVAE(quantizer_type="vq_vae", codebook_size=codebook_size,
                                 latent_dim=latent_dim).to(device)
            print("Training VQ-VAE")

        case "ddcl":
            model = QuantizedVAE(quantizer_type="ddcl", delta=ddcl_delta, latent_dim=latent_dim).to(device)
            print("Training DDCL-VAE")
            print(f"Quantization Delta: {ddcl_delta}")

        case "autoencoder":
            model = QuantizedVAE(quantizer_type="autoencoder", latent_dim=latent_dim).to(device)
            print("Training Autoencoder")

        case _:
            raise ValueError(f"Unknown quantizer_type: {quantizer_type}")

    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 70)

    return model
