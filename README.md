## 📁 Project Structure

```
.
├── train.py           # Main training script
├── models.py          # VAE architecture (Encoder, Decoder, QuantizedVAE)
├── quantizers.py      # Quantization methods (FSQ, DDCL)
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

```bash
python train.py
```

By default, this trains a DDCL-VAE. To switch quantization methods, edit `train.py`:

```python
QUANTIZER_TYPE = 'fsq'   # or 'ddcl'
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
- Reconstruction loss (MSE)
- Regularization loss (DDCL only)
- Codebook usage statistics (FSQ only)

## ️Configuration

Edit the configuration section in `train.py`:

```python
# Training hyperparameters
batch_size = 128
epochs = 50
lr = 3e-4

# FSQ settings
fsq_levels = [8, 6, 6, 5]

# DDCL settings
ddcl_delta = 1 / 15
ddcl_comm_weight = 1e-3
```

## Key Files Explained

### `quantizers.py`
Contains quantization implementations:
- `FSQWrapper`: Wraps the FSQ quantizer with unified interface
- `DDCL_Bottleneck`: Implements DDCL quantization

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