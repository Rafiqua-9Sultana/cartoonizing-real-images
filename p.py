# cartoon_gan.py
"""
Cartoon-style GAN inspired by:
'GAN: A Novel Approach for Cartoonizing Real Images' (paper used for architecture + losses).
See: uploaded PDF. 1
"""

import os
import math
from pathlib import Path
import itertools
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, utils
import torch.nn.functional as F
from torchvision.utils import save_image

# ---------------------------
# Config / Hyperparameters
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256
BATCH_SIZE = 2  # Reduced for testing
LR = 2e-4
BETA1, BETA2 = 0.5, 0.999
EPOCHS = 10  # Increased for better training
CHECKPOINT_DIR = "./checkpoints"
SAMPLE_DIR = "./samples"
DATA_ROOT = "./data/real"   # folder with input real images
CARTOON_ROOT = None         # if you have paired cartoons, else None (we train adversarially)
N_RES_BLOCKS = 4  # Reduced for faster testing
LAMBDA_PERCEPT = 1.0
LAMBDA_EDGE = 5.0
LAMBDA_SPARSE = 1e-4
SAVE_EVERY = 2

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

# ---------------------------
# Dataset
# ---------------------------
class ImageFolderDataset(Dataset):
    def __init__(self, root, transform):
        self.root = Path(root)
        self.paths = sorted([p for p in self.root.rglob("*.jpg")] + [p for p in self.root.rglob("*.png")])
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)

# transforms: resize to IMG_SIZE, convert to tensor, normalize to [-1,1]
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # maps [0,1] -> [-1,1]
])

# ---------------------------
# Utility: init weights
# ---------------------------
def init_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)

# ---------------------------
# Generator
# ---------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(dim, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(dim, affine=True),
        )
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    """
    Downsample -> Residual blocks -> Upsample
    Final activation: tanh (outputs in [-1,1])
    """
    def __init__(self, in_ch=3, out_ch=3, ngf=64, n_res=N_RES_BLOCKS):
        super().__init__()
        # initial conv
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(ngf, affine=True),
            nn.ReLU(True)
        ]
        # downsampling
        curr_dim = ngf
        for _ in range(2):
            model += [
                nn.Conv2d(curr_dim, curr_dim*2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(curr_dim*2, affine=True),
                nn.ReLU(True)
            ]
            curr_dim *= 2
        # residual blocks
        for _ in range(n_res):
            model += [ResidualBlock(curr_dim)]
        # upsampling
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(curr_dim//2, affine=True),
                nn.ReLU(True)
            ]
            curr_dim //= 2
        # final
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(curr_dim, out_ch, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# ---------------------------
# PatchGAN Discriminator
# ---------------------------
class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator: classifies NxN patches as real/fake.
    """
    def __init__(self, in_ch=3, ndf=64):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        for n in range(1, 4):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            layers += [
                nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size=4, stride=2 if n<3 else 1, padding=1),
                nn.InstanceNorm2d(ndf*nf_mult, affine=True),
                nn.LeakyReLU(0.2, True)
            ]
        layers += [nn.Conv2d(ndf*nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ---------------------------
# Perceptual (VGG) loss
# ---------------------------
class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer_ids=(0,5,10,19)):  # conv1_1, conv2_1, conv3_1, conv4_1 indexes (approx)
        super().__init__()
        vgg = models.vgg19(weights='DEFAULT').features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg.to(DEVICE)
        self.layer_ids = layer_ids
        # normalization for VGG input (expects [0,1] normalized with imagenet mean/std)
        self.register_buffer('mean', torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))

    def forward(self, x, y):
        # x,y: expected in [-1,1] -> convert to [0,1] then normalize to imagenet
        def prep(img):
            img = (img + 1.0) / 2.0
            img = (img - self.mean) / self.std
            return img
        x = prep(x)
        y = prep(y)
        loss = 0.0
        xi = x
        yi = y
        for i, layer in enumerate(self.vgg):
            xi = layer(xi)
            yi = layer(yi)
            if i in self.layer_ids:
                loss += F.l1_loss(xi, yi)
        return loss

# ---------------------------
# Edge-promoting loss (Sobel)
# ---------------------------
def sobel_edge_map(img):
    # img: [-1,1], batch x 3 x H x W
    # convert to grayscale
    img_gray = 0.2989*img[:,0:1,:,:] + 0.5870*img[:,1:2,:,:] + 0.1140*img[:,2:3,:,:]
    # Sobel kernels
    kernel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=img.device).view(1,1,3,3)
    kernel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device=img.device).view(1,1,3,3)
    gx = F.conv2d(img_gray, kernel_x, padding=1)
    gy = F.conv2d(img_gray, kernel_y, padding=1)
    edge = torch.sqrt(gx*gx + gy*gy + 1e-6)
    return edge

# ---------------------------
# Sparse regularization on feature maps
# ---------------------------
def sparse_feature_loss(feature_map):
    # L1 on activations encourages sparsity
    return torch.mean(torch.abs(feature_map))

# ---------------------------
# Training loop
# ---------------------------
def train():
    # dataset
    dataset = ImageFolderDataset(DATA_ROOT, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)  # Reduced workers for Windows

    # models
    G = Generator().to(DEVICE)
    D = PatchDiscriminator().to(DEVICE)
    init_weights(G)
    init_weights(D)

    # losses & perceptual
    adversarial_loss = nn.BCEWithLogitsLoss().to(DEVICE)  # discr outputs raw logits
    perceptual = VGGPerceptualLoss().to(DEVICE)

    # optimizers
    opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(BETA1, BETA2))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(BETA1, BETA2))

    fixed_noise = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)  # if using image-to-image, noise could be actual images
    step = 0

    for epoch in range(1, EPOCHS+1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for real in pbar:
            real = real.to(DEVICE)  # real images
            bs = real.size(0)

            # -----------------------
            # Train Discriminator
            # -----------------------
            opt_D.zero_grad()
            # generate fake (G maps real->cartoon style). In paper generator maps noise->cartoon, but image-to-image approach uses real input.
            fake = G(real)  # using real as input conditioned mapping (like conditional GAN / paper mentions A->B)
            # real label (1s) and fake label (0s) patch outputs
            real_logits = D(real)
            fake_logits = D(fake.detach())

            real_targets = torch.ones_like(real_logits, device=DEVICE)
            fake_targets = torch.zeros_like(fake_logits, device=DEVICE)

            d_loss_real = adversarial_loss(real_logits, real_targets)
            d_loss_fake = adversarial_loss(fake_logits, fake_targets)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            d_loss.backward()
            opt_D.step()

            # -----------------------
            # Train Generator
            # -----------------------
            opt_G.zero_grad()

            # adversarial loss (want discriminator to predict real for fake)
            fake_logits_for_g = D(fake)
            g_adv_loss = adversarial_loss(fake_logits_for_g, real_targets)

            # perceptual/content loss
            g_perc_loss = perceptual(fake, real) * LAMBDA_PERCEPT

            # edge-promoting loss: encourage edges in generated to match edges in target (if paired)
            # If unpaired, we encourage generated to have stronger / stylized edges => match edges of an edge-extracted target or use edge magnitude regularizer.
            real_edges = sobel_edge_map(real)
            fake_edges = sobel_edge_map(fake)
            g_edge_loss = F.l1_loss(fake_edges, real_edges) * LAMBDA_EDGE

            # sparse regularization on feature maps - we take a mid-layer activation from generator (approx)
            # For simplicity, apply to fake (encourage sparse output activations)
            g_sparse = sparse_feature_loss(fake) * LAMBDA_SPARSE

            g_loss = g_adv_loss + g_perc_loss + g_edge_loss + g_sparse
            g_loss.backward()
            opt_G.step()

            step += 1
            pbar.set_postfix({
                "d_loss": d_loss.item(),
                "g_adv": g_adv_loss.item(),
                "g_perc": g_perc_loss.item(),
                "g_edge": g_edge_loss.item()
            })

            # Save sample occasionally
            if step % 200 == 0:
                sample_images(G, real, step)

        # epoch end save
        if epoch % SAVE_EVERY == 0:
            torch.save({
                'epoch': epoch,
                'G_state': G.state_dict(),
                'D_state': D.state_dict(),
                'opt_G': opt_G.state_dict(),
                'opt_D': opt_D.state_dict()
            }, os.path.join(CHECKPOINT_DIR, f"ckpt_epoch_{epoch}.pt"))
            # sample
            sample_images(G, real, f"epoch{epoch}")

    print("Training finished.")

# ---------------------------
# Sampling utility
# ---------------------------
def sample_images(G, real_batch, step):
    G.eval()
    with torch.no_grad():
        real = real_batch[:min(8, real_batch.size(0))].to(DEVICE)
        fake = G(real)
        # denormalize to [0,1]
        def denorm(x):
            return (x + 1) / 2
        grid = torch.cat([denorm(real), denorm(fake)], dim=0)
        save_image(grid, os.path.join(SAMPLE_DIR, f"{step}_samples.png"), nrow=8, normalize=False)
    G.train()

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    print("Device:", DEVICE)
    if not os.path.exists(DATA_ROOT) or len(list(Path(DATA_ROOT).rglob("*"))) == 0:
        print(f"ERROR: No data found in {DATA_ROOT}. Put training images there (jpg/png).")
    else:
        train()