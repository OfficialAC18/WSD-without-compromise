from abc import ABC, abstractmethod

class DisentangledDataset(ABC):
    @abstractmethod
    def num_factors(self):
        pass

    @abstractmethod
    def sample_latent_factors(self, random_state, num_samples):
        pass

    @abstractmethod
    def sample_observations_from_factors(self, factors, random_state):
        pass

    @abstractmethod
    def num_examples(self):
        pass