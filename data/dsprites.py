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

    def __init__(self, data_dir, observed_latent_factor_indices=None, seed=42):
        self.data = np.load(os.path.join(data_dir,'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'),
                            encoding='latin1',
                            allow_pickle=True)     
        self.observed_latent_factor_indices = list(range(6)) if observed_latent_factor_indices is None else observed_latent_factor_indices
        self.images = torch.from_numpy(self.data['imgs'])
        self.latents_sizes = torch.from_numpy(self.data['metadata'][()]['latents_sizes']) #[ 1,  3,  6, 40, 32, 32], basically the number of unique values for each factor
        self.unobserved_latent_factor_indices = [i for i in range(len(self.latents_sizes)) if i not in self.observed_latent_factor_indices]
        self.data_shape = self.images[0].unsqueeze(0).shape #(1, 64, 64)
        #[737280, 245760,  40960,   1024,     32,      1], This is the number of indices you need to shift to get to the next unique value of a factor
        self.factor_bases = torch.prod(self.latents_sizes)/torch.cumprod(self.latents_sizes, axis = 0)
        self.rand_generator = torch.Generator().manual_seed(seed)

    @property #basically, allows to access the method without using parentheses
    def num_observed_latent_factors(self):
        return len(self.observed_latent_factor_indices)
     
    @property
    def latent_factor_sizes(self):
        return [self.latents_sizes[i] for i in self.observed_latent_factor_indices]
    
    @property
    def example_shape(self):
        return self.data_shape
    
    @property
    def num_examples(self):
        return len(self.images)
    

    def sample_latent_factors(self, num):
        """
        Sample a batch of latent factors, Y'
        Args:
            num: int, number of examples to generate
        
        Returns:
            factors: torch.Tensor, The set of latent factors
        """
        factors = torch.rand(num, len(self.observed_latent_factor_indices), generator=self.rand_generator)
        factors = (factors*torch.index_select(self.latents_sizes, 0, torch.Tensor(self.observed_latent_factor_indices))).int().floor()
        return factors
    
    def sample_full_latent_vector(self, observed_latent_factors):
        """
        Sample a batch of remaining latent factors, Y-Y', based on the the latent factors Y'
        and combine them to form the full latent vector Y
        Args:
            latent_factors: torch.Tensor, The set of latent factors
        
        Returns:
            all_factors: torch.Tensor, The set of full latent factors, Y' U (Y-Y')
        """
        num_samples = observed_latent_factors.shape[0]
        all_factors = torch.zeros(num_samples, len(self.latents_sizes))
        all_factors[:,self.observed_latent_factor_indices] = observed_latent_factors
        
        if len(self.unobserved_latent_factor_indices) > 0:
            all_factors[:,self.unobserved_latent_factor_indices] = torch.rand(num_samples, len(self.unobserved_latent_factor_indices), generator=self.rand_generator)
            all_factors[:,self.unobserved_latent_factor_indices] = (all_factors[:,self.unobserved_latent_factor_indices]*torch.index_select(self.latents_sizes, 0, torch.Tensor(self.unobserved_latent_factor_indices))).int().floor()
        
        #Final sanity check
        assert torch.all(all_factors[:,self.observed_latent_factor_indices] == observed_latent_factors), "The observed latent factors are not correctly placed"
        
        return all_factors

    def sample_observations(self, num,
                            observed_factors = None, 
                            return_factors=True):
        """
        Attain a set of observations from the dataset
        Args:
            num: int, number of examples to generate
            (Optional) observed_factors: torch.Tensor, the latent factors that we observe
            return_factors: bool, whether to return the latent factors as well
        
        Returns:
            (X,Y): torch.Tensor, The set of examples and their corresponding latent factors
        """
        #Get the subset of latent factors that we observe
        if observed_factors is None:
            all_factors = self.sample_full_latent_vector(self.sample_latent_factors(num))
        else:
            all_factors = self.sample_full_latent_vector(observed_factors)

        images = self.images[torch.matmul(all_factors.int(), self.factor_bases.int())]

        return images, all_factors if return_factors else images

    #Need to test out this function
    def sample_paired_observations_from_factors(self, num=1, k = -1, observed_idx='constant', return_factors=False):
        """
        Generate the required paired examples for Weak Disentanglement
        Args:
            num: int, number of examples to generate
            k: int, number of uncommon factors betweem X1 and X2, if set to -1, it is randomly sampled between 0 and len(observed_latent_factor_indices)
            return_latents: bool, whether to return the latent factors as well
        
        Returns:
            (X1,X2): torch.Tensor, The pair of examples which have d-k common factors.
            Y: torch.Tensor, The label for the pair of examples, This is the latent factor where the examples differ.
            latents: torch.Tensor, the corresponding latent factors

        """
        #Get a set of common factors for the number of pairs
        common_factors = self.sample_latent_factors(num)

        #The set of X1 and X2 examples, this allows us to generate a batch in one go.
        all_factors_x1 = self.sample_full_latent_vector(common_factors)
        all_factors_x2 = self.sample_full_latent_vector(common_factors)

        #Now we determine the k-different factors (Might move this block to a seperate function)
        if k == -1:
            if observed_idx == 'constant':
                k_observed = torch.randint(0, len(self.observed_latent_factor_indices), (1,), generator=self.rand_generator)
            else:
                k_observed = torch.randint(0, len(self.observed_latent_factor_indices), (num,), generator=self.rand_generator)
        else:
            k_observed = torch.tensor(k)

        #Since assumption is based on the fact that only a single factor is different
        labels = []
        
        #Randomly sample the indices of the k-different factors, remember that this is for the observed latent factors
        observed_indices = torch.Tensor(self.observed_latent_factor_indices)
        if k_observed == 'constant':
            diff_factors = torch.stack([observed_indices[torch.randperm(len(observed_indices), generator=self.rand_generator)][:k_observed]] for _ in range(num))
        else:
            diff_factors = torch.stack([observed_indices[torch.randperm(len(observed_indices), generator=self.rand_generator)][:k_observed[idx_inner]]] for idx_inner in range(num))
    
        for idx in range(num):
            for i in diff_factors[idx]:
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


