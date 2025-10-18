## Project Structure

```
.
├── train.py           # Main training script
├── models.py          # VAE architecture (Encoder, Decoder, QuantizedVAE)
├── quantizers.py      # Quantization methods (FSQ, DDCL, VAE, VQ-VAE)
├── utils.py           # Visualization and utilities
├── outputs/           # Generated visualizations (created automatically)
├── checkpoints/       # Saved model checkpoints (created automatically)
└── data/             # CIFAR-10 dataset (downloaded automatically)
```

### Installation

```bash
pip install torch torchvision vector-quantize-pytorch tqdm matplotlib
```

### Training

**Basic usage**:
```bash
python train.py --quantizer_type fsq  # Options: fsq, ddcl, vae, vq_vae
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
- Codebook usage statistics (FSQ only)

## Configuration

**Available flags**:
```bash
--quantizer_type {fsq,ddcl,vae,vq_vae}  # Quantization method
--batch_size 16                          # Batch size
--epochs 100                             # Training epochs
--lr 0.001                               # Learning rate

# FSQ specific
--fsq_levels 8 8 8 8                     # FSQ quantization levels

# DDCL specific
--ddcl_delta 0.1                         # DDCL grid width

# VQ-VAE specific
--codebook_size 128                      # Codebook size

# VAE/VQ-VAE/DDCL
--reg_loss_weight 1e-4                   # KL (VAE), commitment (VQ-VAE), communication (DDCL)

# WandB
--use_wandb {true,false}                 # Enable logging
--wandb_project ddcl-vae                 # Project name
```

## Key Files Explained

### `quantizers.py`
Quantization implementations:
- `FSQWrapper`: FSQ quantizer with unified interface
- `DDCL_Bottleneck`: DDCL quantization
- `VanillaVAE`: Gaussian VAE with KL divergence
- `VQVAEQuantizer`: Vector quantization with codebook

### `models.py`
Contains network architectures:
- `Encoder`: CNN encoder (32x32 → 4x4 latent)
- `Decoder`: CNN decoder (4x4 latent → 32x32)
- `QuantizedVAE`: Main model with pluggable quantizers

### `utils.py`
Helper functions:
- `visualize_reconstructions()`: Generate 4x4 comparison grids
- `compute_codebook_usage()`: Analyze FSQ codebook utilization
- `save_checkpoint()` / `load_checkpoint()`: Model persistence

## Understanding FSQ Levels

FSQ `levels` parameter defines discrete values per latent dimension:

```python
levels = [8, 5, 5, 5]
# Dim 0: 8 values   → {-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5}
# Dim 1: 5 values   → {-2, -1, 0, 1, 2}
# Dim 2: 5 values   → {-2, -1, 0, 1, 2}
# Dim 3: 5 values   → {-2, -1, 0, 1, 2}
# Total codebook size = 8 × 5 × 5 × 5 = 1000
```