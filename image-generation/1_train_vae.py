import torch
from image_generation.vqgan_vae import VQGanVAE, DDCLGanVAE, FSQGanVAE
from image_generation.trainers import VQGanVAETrainer

vae = VQGanVAE(
    dim=128,
    codebook_size=1024,
    layers=5,
    lookup_free_quantization=False,
    use_hinge_loss=False,
    encdec_layer_mults=(1, 1, 2, 2, 4),
    encdec_num_resnet_blocks=2,
    vq_codebook_dim=256,
    vq_commitment_weight=0.25,
)


trainer = VQGanVAETrainer(
    vae=vae,
    image_size=256,
    folder="/home/eigy/Data/Silo/rl/ddcl-applications/image-generation/test",
    batch_size=4,
    grad_accum_every=64,
    num_train_steps=1_000_000,
    save_model_every=10_000,
    save_results_every=10_000,
    lr=1e-4,
).cuda()

trainer.train()
