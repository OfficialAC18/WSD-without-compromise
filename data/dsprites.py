import os
import torch
import numpy as np
from data.disentangled import DisentangledSampler


class Dsprites(DisentangledSampler):
    """
    Dsprites dataset introduced in the paper, Beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework
    Link to the dataset: https://github.com/deepmind/dsprites-dataset.

    Factors of variation:
    0 - Color (1 possible value)
    1 - Shape (3 possible values)
    2 - Scale (6 possible values)
    3 - Orientation (40 possible values)
    4 - X position (32 possible values)
    5 - Y position (32 possible values) 
    """

    def __init__(self, data_dir, observed_latent_factor_indices=None, seed=42):
        self.data = np.load(os.path.join(data_dir,'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'),
                            encoding='latin1',
                            allow_pickle=True)
        self.images = torch.from_numpy(self.data['imgs'])
        # (1, 64, 64)
        self.data_shape = self.images[0].unsqueeze(0).shape

        # [ 1,  3,  6, 40, 32, 32], the number of unique values for each factor
        self.latent_factor_sizes = torch.from_numpy(self.data['metadata'][()]['latents_sizes'])
        # [737280, 245760, 40960, 1024, 32, 1], the number of indices you need to shift to get to the next unique value of a factor
        self.factor_bases = torch.prod(self.latent_factor_sizes) / torch.cumprod(self.latent_factor_sizes, dim=0)

        # which latent factors are observed and can therefore be sampled
        self.observed_latent_factor_indices = list(
            range(6)) if observed_latent_factor_indices is None else observed_latent_factor_indices
        self.unobserved_latent_factor_indices = [i for i in range(len(self.latent_factor_sizes)) if i not in self.observed_latent_factor_indices]

        self.rand_generator = torch.Generator().manual_seed(seed)

    @property
    def num_observed_latent_factors(self):
        return len(self.observed_latent_factor_indices)
     
    @property
    def observed_latent_factor_sizes(self):
        return [self.latent_factor_sizes[i] for i in self.observed_latent_factor_indices]
    
    @property
    def example_shape(self):
        return self.data_shape
    
    @property
    def num_examples(self):
        return len(self.images)
    

    def sample_latent_factors(self, num_samples):
        """
        Sample a batch of latent factors
        Args:
            num_samples: int, number of examples to generate
        
        Returns:
            factors: torch.Tensor, The set of latent factors
        """
        factors = torch.rand(num_samples, len(self.latent_factor_sizes), generator=self.rand_generator)
        factors = (factors * self.latent_factor_sizes).int().floor()
        return factors

    def sample_observations(self, num_samples,
                            observed_factors = None,
                            return_factors=True):
        """
        Attain a set of observations from the dataset
        Args:
            num_samples: int, number of examples to generate
            observed_factors: torch.Tensor, the latent factors that we observe
            return_factors: bool, whether to return the latent factors as well
        
        Returns:
            X: torch.Tensor, The set of examples
            Y (if return_factors): torch.Tensor, Their corresponding latent factors
        """
        latent_factors = self.sample_latent_factors(num_samples)
        images = self.images[torch.matmul(latent_factors.int(), self.factor_bases.int())]

        if return_factors:
            return images, latent_factors
        else:
            return images

    def sample_paired_observations(self, num_samples=1, k = -1, observed_idx='constant', return_latents=False):
        """
        Generate the required paired examples for Weak Disentanglement
        Args:
            num_samples: int, number of examples to generate
            k: int, number of uncommon factors between X1 and X2. If set to -1, it is randomly sampled between 1 and all observed factors
            observed_idx: str, 'constant' or 'random', if 'constant', then the number of different factors are the same for all pairs, if 'random', then the number of different factors are randomly sampled for each pair
            return_latents: bool, whether to return the latent factors as well
        
        Returns:
            (X1,X2): torch.Tensor, The pair of examples which have d-k common factors.
            labels: torch.Tensor, binary vector representing which factors are different
            all_factors_x1, all_factors_x2: torch.Tensor, the corresponding latent factors

        """
        # The set of X1 and X2 examples, initially the same, as we have not applied variation yet.
        all_factors_x1 = self.sample_latent_factors(num_samples)
        all_factors_x2 = all_factors_x1.clone()

        # We determine how many factors each sample gets changed
        if k == -1:
            # At least one changes, at least one stays the same
            if observed_idx == 'constant':
                k_observed = torch.randint(1, len(self.observed_latent_factor_indices), (1,), generator=self.rand_generator).repeat(num_samples,)
            else:
                k_observed = torch.randint(1, len(self.observed_latent_factor_indices), (num_samples,), generator=self.rand_generator)
        else:
            k_observed = torch.tensor([k]).repeat(num_samples,)
        
        #Randomly sample the indices of the k-different factors, remember that this is for the observed latent factors
        observed_indices = torch.Tensor(self.observed_latent_factor_indices)
        diff_factors = [observed_indices[torch.randperm(len(observed_indices), generator=self.rand_generator)][:k_observed[idx_inner]] for idx_inner in range(num_samples)]

        labels = torch.zeros(all_factors_x1.shape)
        for idx in range(num_samples):
            for i in diff_factors[idx]:
                all_factors_x2[idx,i.int()] = torch.randint(0, self.latent_factor_sizes[i.int()], (1,), generator=self.rand_generator)
                labels[idx,i.int()] = 1

        #Get the corresponding images
        images_x1 = self.images[torch.matmul(all_factors_x1.int(), self.factor_bases.int())]
        images_x2 = self.images[torch.matmul(all_factors_x2.int(), self.factor_bases.int())]

        #Concatenate the images across the first IMAGE dimension (Image shape should be (n, 1, 128, 64)), where the second entry is the channel

        image_pairs = torch.concatenate((images_x1, images_x2), dim=1).unsqueeze(1).float()

        if return_latents:
            return image_pairs, labels, all_factors_x1, all_factors_x2
        else:
            return image_pairs, labels


