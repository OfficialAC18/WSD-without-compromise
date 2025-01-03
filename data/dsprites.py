import os
import torch
import numpy as np
from disentangled import DisentangledSampler


class Dsprites(DisentangledSampler):
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
        self.latents_sizes = torch.from_numpy(self.data['metadata'][()]['latents_sizes'])
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
    
    @property
    def num_examples(self):
        return len(self.images)
    

    def _sample_latent_factors(num,self):
        """Sample a batch of latent factors, Y"""
        factors = torch.rand(num, self.latent_factor_indices, generator=self.rand_generator)
        factors = (factors*torch.index_select(self.latents_sizes, 0, torch.Tensor(self.latent_factor_indices))).floor().long()
        return factors
    

    def sample_observations(self, num, return_factors=True):
        """
        Attain a set of observations from the dataset
        Args:
            num: int, number of examples to generate
            return_factors: bool, whether to return the latent factors as well
        
        Returns:
            (X,Y): torch.Tensor, The set of examples and their corresponding latent factors
        """
        factors = self._sample_latent_factors(num)
        all_factors = torch.zeros(num, len(self.latents_sizes))

        rem_factors = torch.randn(num, len(self.observation_factor_indices), generator=self.rand_generator)
        rem_factors = (rem_factors*torch.index_select(self.latents_sizes, 0, self.observation_factor_indices)).floor().long()

        all_factors[:,self.latent_factor_indices] = factors
        all_factors[:,self.observation_factor_indices] = rem_factors

        images = self.images[torch.matmul(all_factors.int(), self.factor_bases.int())]

        return images, factors if return_factors else images

    def sample_observations_from_factors(self, factors, random_state):
        """
        Attain a set of observations from the dataset given the latent factors
        Args:
            factors: torch.Tensor, the latent factors
            random_state: int, the random state for the generator
        
        Returns:
            X: torch.Tensor, The set of examples
        """

    #Need to test out this function
    def sample_paired_observations_from_factors(self, num, k = -1, observed_idx='constant', return_factors=False):
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

        #Now we determine the k-different factors (Might move this block to a seperate function)
        if k == -1:
            if observed_idx == 'constant':
                k_observed = torch.randint(0, len(self.latent_factor_indices), (1,), generator=self.rand_generator)
            else:
                k_observed = torch.randint(0, len(self.latent_factor_indices), (num,), generator=self.rand_generator)
        else:
            k_observed = torch.tensor(k)

        #Since assumption is based on the fact that only a single factor is different
        labels = []
        
        #Randomly sample the indices of the k-different factors
        if k_observed == 'constant':
            diff_factors = torch.randint(0, len(self.latent_factor_indices), (num, k_observed), generator=self.rand_generator)
            for idx in range(num):
                for i in diff_factors:
                    all_factors_x2[idx,i] = torch.randint(0, self.latents_sizes[i], (1,), generator=self.rand_generator)
                labels.append(i)
        else:
            for idx in range(num):
                diff_factors = torch.randint(0, len(self.latent_factor_indices), (1, k_observed[idx]), generator=self.rand_generator)
                for i in diff_factors:
                    all_factors_x2[idx,i] = torch.randint(0, self.latents_sizes[i], (1,), generator=self.rand_generator)
                labels.append(i)   

        #Get the corresponding images
        images_x1 = self.images[torch.matmul(all_factors_x1.int(), self.factor_bases.int())]
        images_x2 = self.images[torch.matmul(all_factors_x2.int(), self.factor_bases.int())]

        #Concatenate the images across the first IMAGE dimension (Image shape should be (n, 1, 128, 64))
        image_pairs = torch.concatenate((images_x1, images_x2), dim=1).unsqueeze(1)
        #Convert the labels to a tensor
        labels = torch.tensor(labels)

        return image_pairs, labels, all_factors_x1, all_factors_x2 if return_factors else image_pairs, labels


