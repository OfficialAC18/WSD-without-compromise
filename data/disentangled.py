from torch.utils.data import Dataset
from abc import ABC, abstractmethod


class DisentangledSampler(ABC):
    @abstractmethod
    def num_observed_latent_factors(self):
        raise NotImplementedError()
    
    @abstractmethod
    def observed_latent_factor_sizes(self):
        raise NotImplementedError()
    
    @abstractmethod
    def example_shape(self):
        raise NotImplementedError()
    
    @abstractmethod
    def sample_latent_factors(self, num_samples):
        raise NotImplementedError()
    
    @abstractmethod
    def sample_observations(self, num, observed_factors = None, return_factors = True):
        raise NotImplementedError()

    @abstractmethod
    def num_examples(self):
        raise NotImplementedError()



class DisentangledDataset(Dataset):
    def __init__(self, sampler, observed_idx='constant', k_observed = 1, return_latents=False):
        self.sampler = sampler
        self.observed_idx = observed_idx
        self.k_observed = k_observed
        self.return_latents = return_latents

    def __len__(self):
        return self.sampler.num_examples

    def __getitem__(self, idx):
        return self.sampler.sample_paired_observations(num_samples = 1,
                                                       k = self.k_observed,
                                                       observed_idx = self.observed_idx,
                                                       return_latents = self.return_latents)

    def __getitems__(self, idx):
        return self.sampler.sample_paired_observations(num_samples = len(idx),
                                                       k = self.k_observed,
                                                       observed_idx = self.observed_idx,
                                                       return_latents = self.return_latents)