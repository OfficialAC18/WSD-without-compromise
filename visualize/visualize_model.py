from data.dsprites import Dsprites
from models.GVAE import GroupVAELabels, GroupVAEArgMax
from models.ML_VAE import MLVAELabels, MLVAEArgMax
from torchvision.utils import make_grid, save_image

import torch
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

def visualize_model(seed, model_name, aggregate, latent_dim):
    filename = f"vis_model_{seed}.png"
    dsprites = Dsprites(current_dir + '/../datasets/dSprites')
    images, labels = dsprites.sample_paired_observations(num_samples=9)

    if model_name == 'G_VAE':
        if aggregate == 'label':
            model = GroupVAELabels(data_shape=torch.Size([1, 64, 64]), latent_dim=latent_dim, labels=True)
        else:
            model = GroupVAEArgMax(data_shape=torch.Size([1, 64, 64]), latent_dim=latent_dim)
    else:
        if aggregate == 'label':
            model = MLVAELabels(data_shape=torch.Size([1, 64, 64]), latent_dim=latent_dim, labels=True)
        else:
            model = MLVAEArgMax(data_shape=torch.Size([1, 64, 64]), latent_dim=latent_dim)
    model.load_state_dict(torch.load(current_dir + f'/../trained_models/trained_model_{seed}.pth'))
    reconstructed_tuple = model(images, labels)
    reconstructed_images = torch.cat((reconstructed_tuple[0], reconstructed_tuple[1]), dim=2)
    reconstructed_images = torch.nn.Sigmoid()(reconstructed_images)
    comparison = torch.cat([images, reconstructed_images], dim=3)
    grid = make_grid(comparison, nrow=3, padding=4, pad_value=1.0)
    save_image(grid, current_dir + f'/{filename}')
    print(f"Visualization saved as {filename}")

if __name__=='__main__':
    visualize_model(70008, 'ML_VAE', 'label', 6)