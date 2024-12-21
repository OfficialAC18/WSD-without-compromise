import os
import torch
import numpy as np
from disentangled import DisentangledDataset


class Dsprites(DisentangledDataset):
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

    def __init__(self, data_dir, latent_factor_indices=None, seed=42):
        self.data = np.load(os.path.join(data_dir,'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'))     
        self.latent_factor_indices = list(range(6)) if latent_factor_indices is None else latent_factor_indices
        self.images = torch.from_numpy(self.data['imgs'])
        self.latents_sizes = self.data['metadata'][()]['latents_sizes']
        self.observation_factor_indices = [i for i in range(len(self.latents_sizes)) if i not in self.latent_factor_indices]
        self.data_shape = self.images[0].unsqueeze(0).shape
        self.factor_bases = torch.prod(torch.Tensor(self.latents_sizes))/torch.cumprod(torch.Tensor(self.latents_sizes)).item()
        self.rand_generator = torch.Generator().manual_seed(seed)

    @property #basically, allows to access the method without using parentheses
    def num_factors(self):
        return len(self.latent_factor_indices)
     
    @property
    def latent_factor_sizes(self):
        return [self.latents_sizes[i] for i in self.latent_factor_indices]
    
    @property
    def example_shape(self):
        return self.data_shape
    

    def _sample_latent_factors(num,self):
        """Sample a batch of latent factors, Y"""
        factors = torch.rand(num, self.latent_factor_indices, generator=self.rand_generator)
        factors = (factors*torch.index_select(self.latents_sizes, 0, torch.Tensor(self.latent_factor_indices))).floor().long()
        return factors
    

    #Need to test out this function
    def sample_observations_from_factors(self, num, k = -1, return_latents=False):
        """
        Generate the required paired examples for Weak Disentanglement
        Args:
            num: int, number of examples to generate
            k: int, number of uncommon factors betweem X1 and X2, if set to -1, it is randomly sampled between 0 and len(latent_factor_indices)
            return_latents: bool, whether to return the latent factors as well
        
        Returns:
            (X1,X2): torch.Tensor, The pair of examples which have d-k common factors.
            Y: torch.Tensor, The label for the pair of examples, This is the latent factor where the examples differ.
            latents: torch.Tensor, the corresponding latent factors

        """
        #Get a set of common factors for the number of pairs
        common_factors = self._sample_latent_factors(num)

        #The set of X1 and X2 examples, this allows us to generate a batch in one go.
        all_factors_x1 = torch.zeros(common_factors.shape[0], len(self.latents_sizes))
        all_factors_x2 = torch.zeros(common_factors.shape[0], len(self.latents_sizes))

        #Generate the uncommon factors for X1 and X2 examples (This is generally the noise variables, NOT the k-different vars)
        if len(self.latent_factor_indices) != len(self.latents_sizes):
            rem_factors_x1 = torch.randn(common_factors.shape[0], len(self.observation_factor_indices), generator=self.rand_generator)
            rem_factors_x1 = (rem_factors_x1*torch.index_select(self.latents_sizes, 0, torch.Tensor(self.observation_factor_indices))).floor().long()

            rem_factors_x2 = torch.randn(common_factors.shape[0], len(self.observation_factor_indices), generator=self.rand_generator)
            rem_factors_x2 = (rem_factors_x2*torch.index_select(self.latents_sizes, 0, torch.Tensor(self.observation_factor_indices))).floor().long()
            
            all_factors_x1[:,self.observation_factor_indices] = rem_factors_x1
            all_factors_x2[:,self.observation_factor_indices] = rem_factors_x2


        #Assemble the all_factors tensor accordingly
        all_factors_x1[:,self.latent_factor_indices] = common_factors
        all_factors_x2[:,self.latent_factor_indices] = common_factors

        #Now we determine the k-different factors
        if k == -1:
            k_observed = torch.randint(0, len(self.latent_factor_indices), (num,), generator=self.rand_generator)
        else:
            k_observed = torch.tensor(k)
        
        #Randomly sample the indices of the k-different factors
        for idx in range(num):
            diff_factors = torch.randint(0, len(self.latent_factor_indices), (k_observed[idx],), generator=self.rand_generator)
            for i in diff_factors:
                all_factors_x2[idx,i] = torch.randint(0, self.latents_sizes[i], (1,), generator=self.rand_generator)

        return all_factors
