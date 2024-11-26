import os
import torch
import numpy as np
from torch.utils.data import Dataset




#Need to make sure that
#a seed propagates to the torch.rand function
class Dsprites(Dataset):
    """
    Dsprites dataset introudced in the paper, Beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework
    Link to the dataset: https://github.com/deepmind/dsprites-dataset.

    Factors of variation:
    0 - Color (1 possible value)
    1 - Shape (3 possible values)
    2 - Scale (6 possible values)
    3 - Orientation (40 possible values)
    4 - X position (32 possible values)
    5 - Y position (32 possible values) 
    """

    def __init__(self, data_dir, latent_factor_indices=None):
        self.data = np.load(os.path.join(data_dir,'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'))     
        self.latent_factor_indices = list(range(6)) if latent_factor_indices is None else latent_factor_indices
        self.images = torch.from_numpy(self.data['imgs'])
        self.latents_sizes = self.data['metadata'][()]['latents_sizes']
        self.observation_factor_indices = [i for i in range(len(self.latents_sizes)) if i not in self.latent_factor_indices]
        self.data_shape = self.images[0].unsqueeze(0).shape
        self.factor_bases = torch.prod(torch.Tensor(self.latents_sizes))/torch.cumprod(torch.Tensor(self.latents_sizes)).item()
        

    @property #basically, allows to access the method without using parentheses
    def num_factors(self):
        return len(self.latent_factor_indices)
    
    @property
    def latent_factor_sizes(self):
        return [self.latents_sizes[i] for i in self.latent_factor_indices]
    
    @property
    def example_shape(self):
        return self.data_shape
    

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.sample_latent_factors(idx)


    def sample_latent_factors(self, idx):
        """Sample a batch of latent factors, Y"""
        factors = torch.rand(len(idx), self.latent_factor_indices)
        factors = (factors*torch.index_select(self.factor_bases, 0, torch.Tensor(self.latent_factor_indices))).floor().long()
        return factors
    

    #Need to test out this function
    def sample_observations_from_factors(self, factors):
        """Sample remaining factors based on the sampled latent factors"""
        all_factors = torch.zeros(len(factors.shape[0]), len(self.latents_sizes))
        rem_factors = torch.randn(len(factors.shape[0]), len(self.observation_factor_indices))
        rem_factors = (rem_factors*torch.index_select(self.factor_bases, 0, torch.Tensor(self.observation_factor_indices))).floor().long()

        #Assemble the all_factors tensor accordingly
        all_factors[:,self.latent_factor_indices] = factors
        all_factors[:,self.observation_factor_indices] = rem_factors

        return all_factors 
