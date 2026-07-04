from data.dsprites import Dsprites
from models.GVAE import GroupVAELabels, GroupVAEArgMax
from models.ML_VAE import MLVAELabels, MLVAEArgMax
from torchvision.utils import make_grid, save_image

import torch
import os
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))

def visualize_model(args, step, sampler, model, device):
    filename = f"vis_model_{args.seed}_{step}.png"
    images, labels = sampler.sample_paired_observations(num_samples=9)
    images = images.to(device)
    labels = labels.to(device)

    if args.model == 'VAE':
        recon1 = model(images, labels)[0]
        recon2 = model(images[:, :, 64:, :], labels)[1]
        reconstructed_tuple = (recon1, recon2)
    else:
        reconstructed_tuple = model(images, labels)
    reconstructed_images = torch.cat((reconstructed_tuple[0], reconstructed_tuple[1]), dim=2)
    reconstructed_images = torch.nn.Sigmoid()(reconstructed_images)
    comparison = torch.cat([images, reconstructed_images], dim=3)
    grid = make_grid(comparison, nrow=3, padding=4, pad_value=1.0)
    save_image(grid, current_dir + f'/{filename}')
    logging.info(f"Visualization for step {step} saved as {filename}")