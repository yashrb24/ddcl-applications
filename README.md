## Project Structure

```
.
├── train.py              # Main training script with CLI interface
├── models.py             # VAE architecture (Encoder, Decoder, QuantizedVAE)
├── quantizers.py         # Quantization methods (FSQ, DDCL, VAE, VQ-VAE)
├── dataloading.py        # Data loading utilities for CIFAR-10
├── train_utils.py        # Training and validation functions
├── utils.py              # Visualization and codebook analysis utilities
├── run_sweep.py          # WandB hyperparameter sweep script
├── sweep_config.yaml     # Configuration for hyperparameter sweeps
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Python project configuration
├── outputs/              # Generated visualizations (created automatically)
├── checkpoints/          # Saved model checkpoints (created automatically)
├── data/                 # CIFAR-10 dataset (downloaded automatically)
└── archived/             # Archived old implementations
```

### Installation

```bash
pip install -r requirements.txt
```

### Training

**Basic usage**:
```bash
python train.py --quantizer_type fsq  # Options: fsq, ddcl, vae, vq_vae, autoencoder
```

**Examples**:
```bash
# FSQ with custom levels
python train.py --quantizer_type fsq --fsq_levels 8 8 8 8

# DDCL with custom delta
python train.py --quantizer_type ddcl --ddcl_delta 0.1 --reg_loss_weight 1e-4

# Vanilla VAE
python train.py --quantizer_type vae --reg_loss_weight 1e-4

# VQ-VAE with codebook
python train.py --quantizer_type vq_vae --codebook_size 128 --reg_loss_weight 1e-4

# Vanilla Autoencoder (no quantization)
python train.py --quantizer_type autoencoder --latent_dim 8

# With WandB logging
python train.py --quantizer_type fsq --use_wandb true --wandb_project my-project
```

## Output

### During Training
- **Visualizations**: 4x4 grids saved to `outputs/` after each epoch
  - Top 2 rows: Original images
  - Bottom 2 rows: Reconstructions
  
- **Checkpoints**: Saved to `checkpoints/`
  - Best model: `{quantizer_type}_vae_best.pt`
  - Periodic: `{quantizer_type}_vae_epoch_{n}.pt`

### Metrics Tracked
- Reconstruction loss (MSE for all)
- Regularization loss (KL divergence for VAE, commitment loss for VQ-VAE, communication loss for DDCL)
- Codebook usage statistics (FSQ, VQ-VAE and DDCL only)

## Configuration

**Available flags**:
```bash
--quantizer_type {fsq,ddcl,vae,vq_vae,autoencoder}  # Quantization method
--batch_size 16                                     # Batch size
--epochs 100                                        # Training epochs
--lr 0.001                                          # Learning rate

# FSQ specific
--fsq_levels 8 8 8 8                                # FSQ quantization levels

# DDCL specific
--ddcl_delta 0.1                                    # DDCL grid width

# VQ-VAE specific
--codebook_size 128                                 # Codebook size

# General quantizer settings
--latent_dim 4                                      # Latent space dimensionality (non-FSQ only)

# VAE/VQ-VAE/DDCL
--reg_loss_weight 1e-4                              # KL (VAE), commitment (VQ-VAE), communication (DDCL)

# WandB
--use_wandb {true,false}                            # Enable logging
--wandb_project ddcl-vae                            # Project name
```