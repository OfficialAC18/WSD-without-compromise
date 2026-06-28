from data.dsprites import Dsprites
from models.GVAE import GroupVAELabels, GroupVAEArgMax
from models.ML_VAE import MLVAELabels, MLVAEArgMax
from torchvision.utils import make_grid, save_image

import torch
import os
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))

def visualize_model(args, step):
    filename = f"vis_model_{args.seed}_{step}.png"
    dsprites = Dsprites(current_dir + '/../datasets/dSprites', seed=args.seed)
    images, labels = dsprites.sample_paired_observations(num_samples=9)

    if args.model == 'G_VAE':
        if args.aggregate == 'label':
            model = GroupVAELabels(data_shape=torch.Size([1, 64, 64]), latent_dim=args.latent_dim, labels=True)
        else:
            model = GroupVAEArgMax(data_shape=torch.Size([1, 64, 64]), latent_dim=args.latent_dim)
    elif args.model == 'ML_VAE':
        if args.aggregate == 'label':
            model = MLVAELabels(data_shape=torch.Size([1, 64, 64]), latent_dim=args.latent_dim, labels=True)
        else:
            model = MLVAEArgMax(data_shape=torch.Size([1, 64, 64]), latent_dim=args.latent_dim)
    elif args.model == 'VAE':
        from models.VAE import VAE
        model = VAE(data_shape=torch.Size([1, 64, 64]), latent_dim=args.latent_dim)
    model.load_state_dict(torch.load(current_dir + f'/../trained_models/trained_model_{args.seed}_{step}.pth'))

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