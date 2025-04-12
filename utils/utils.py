import os
from data.dsprites import Dsprites

def get_dataset(dataset_name, seed=42, latent_factor_indices=None):
    DATA_DIR = 'datasets/'
    if dataset_name == 'dSprites':
        return Dsprites(os.path.join(DATA_DIR,'dSprites'), seed=seed, latent_factor_indices=latent_factor_indices)
    else:
        raise ValueError(f'Dataset {dataset_name} not implemented yet')
