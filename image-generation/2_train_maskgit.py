

import torch
from image_generation.vqgan_vae import VQGanVAE
from image_generation.muse_maskgit_pytorch import MaskGit, MaskGitTransformer

# first instantiate your vae

vae = VQGanVAE(
    dim = 64,
    codebook_size = 64,
    layers=4
)

vae.load('/home/eigy/Data/Silo/rl/ddcl-applications/image-generation/results/vae.10.pt') # you will want to load the exponentially moving averaged VAE

# then you plug the vae and transformer into your MaskGit as so

# (1) create your transformer / attention network

transformer = MaskGitTransformer(
    num_tokens = 64,       # must be same as codebook size above
    seq_len = 64,            # must be equivalent to fmap_size ** 2 in vae
    dim = 64,                # model dimension
    depth = 1,                # depth
    dim_head = 64,            # attention head dimension
    heads = 1,                # attention heads,
    ff_mult = 4,              # feedforward expansion factor
    t5_name = 't5-small',     # name of your T5
)

# (2) pass your trained VAE and the base transformer to MaskGit

base_maskgit = MaskGit(
    vae = vae,                 # vqgan vae
    transformer = transformer, # transformer
    image_size = 128,          # image size
    cond_drop_prob = 0.25,     # conditional dropout, for classifier free guidance
).cuda()

# ready your training text and images

texts = [
    'a child screaming at finding a worm within a half-eaten apple',
    'lizard running across the desert on two feet',
    'waking up to a psychedelic landscape',
    'seashells sparkling in the shallow waters'
]

images = torch.randn(4, 3, 128, 128).cuda()

# feed it into your maskgit instance, with return_loss set to True

loss = base_maskgit(
    images,
    texts = texts
)

loss.backward()

# do this for a long time on much data
# then...

images = base_maskgit.generate(texts = [
    'a whale breaching from afar',
    'young girl blowing out candles on her birthday cake',
    'fireworks with blue and green sparkles'
], cond_scale = 3.) # conditioning scale for classifier free guidance

print(images.shape) # (3, 3, 256, 256)