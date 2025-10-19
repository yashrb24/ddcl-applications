import torch
from torch.nn import functional as F
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm


def train_epoch(model, dataloader, optimizer, criterion, device, reg_loss_weight=0.01, compute_perceptual=True):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_reg_loss = 0
    total_perceptual_loss = 0

    # Initialize perceptual loss function only if needed
    perceptual_loss_fn = None
    if compute_perceptual:
        perceptual_loss_fn = LearnedPerceptualImagePatchSimilarity(net_type='vgg', reduction='sum', normalize=True).to(
            device)

    for batch_idx, (data, _) in enumerate(tqdm(dataloader, desc="Training")):
        data = data.to(device)
        optimizer.zero_grad()

        recon, _, reg_loss = model(data)

        recon_loss = criterion(recon, data)

        # Add regularization loss (weighted)
        total_batch_loss = recon_loss + reg_loss_weight * reg_loss

        total_batch_loss.backward()
        optimizer.step()

        # Compute perceptual loss for evaluation only (no gradients)
        if compute_perceptual:
            with torch.no_grad():
                perceptual_loss = perceptual_loss_fn(recon, data)
                total_perceptual_loss += perceptual_loss.item()

        total_loss += total_batch_loss.item()
        total_recon_loss += recon_loss.item()
        total_reg_loss += reg_loss if isinstance(reg_loss, float) else reg_loss.item()

    metrics = {
        "total_loss": total_loss / len(dataloader),
        "recon_loss": total_recon_loss / len(dataloader),
        "reg_loss": total_reg_loss / len(dataloader),
    }

    if compute_perceptual:
        metrics["perceptual_loss"] = total_perceptual_loss / len(dataloader)

    return metrics


@torch.no_grad()
def validate(model, dataloader, device, compute_perceptual=True):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_perceptual_loss = 0

    # Initialize perceptual loss function only if needed
    perceptual_loss_fn = None
    if compute_perceptual:
        perceptual_loss_fn = LearnedPerceptualImagePatchSimilarity(net_type='vgg', reduction='sum', normalize=True).to(
            device)

    for data, _ in dataloader:
        data = data.to(device)
        recon, _, _ = model(data)
        loss = F.mse_loss(recon, data)
        total_loss += loss.item()

        # Compute perceptual loss for evaluation only if needed
        if compute_perceptual:
            perceptual_loss = perceptual_loss_fn(recon, data)
            total_perceptual_loss += perceptual_loss.item()

    avg_recon_loss = total_loss / len(dataloader)
    avg_perceptual_loss = total_perceptual_loss / len(dataloader) if compute_perceptual else None

    return avg_recon_loss, avg_perceptual_loss
