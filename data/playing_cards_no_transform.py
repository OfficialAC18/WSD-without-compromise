import os
import csv
import torch
import torchvision
from data.disentangled import DisentangledSampler



class PlayingCardsNoTransform(DisentangledSampler):
    """
    Playing cards dataset without any lighting or positional transformations.
    This is the simplest version of the dataset we shall be using.

    Factors of variation:
    0 - Card Rank (13 possible values)
    1 - Card Suit (4 possible values)
    
    I am assuming that the latent representations will learn 
    disentangled representations of the card rank, suit and color.

    """

    def __init__(self, data_dir = 'datasets/playing_card_games/simple_playing_cards',
                observed_latent_factor_indices=None,
                seed=42):
        self.data_dir = data_dir
        self.labels = {}
        with open(os.path.join(data_dir, 'simple_playing_card_labels.csv'), 'r') as f:
            reader = csv.reader(f)
            next(reader) #This is to skip the header
            for row in reader:
                self.labels[row[1]] = row[0]
        self.data_shape = torchvision.io.read_image(os.path.join(data_dir, self.labels['ac']))[1:,:,:].shape
        self.latent_sizes = torch.tensor([13, 4])
        self.observed_latent_factor_indices = list(range(2)) if observed_latent_factor_indices is None else observed_latent_factor_indices
        self.unobserved_latent_factor_indices = [i for i in range(len(self.latent_sizes)) if i not in self.observed_latent_factor_indices]
        self.rand_generator = torch.Generator().manual_seed(seed)

        self.latent2rank = {i:f'{i+1}' for i in range(10)}
        self.latent2rank[0] = 'a'
        self.latent2rank[10] = 'j'
        self.latent2rank[11] = 'q'
        self.latent2rank[12] = 'k'

        self.latent2suit = {}
        self.latent2suit[0] = 'c'
        self.latent2suit[1] = 'd'
        self.latent2suit[2] = 'h'
        self.latent2suit[3] = 's'


    @property
    def num_observed_latent_factors(self):
        return len(self.observed_latent_factor_indices)
    
    @property
    def latent_factor_sizes(self):
        return [self.latent_sizes[i] for i in self.observed_latent_factor_indices]
    
    @property
    def example_shape(self):
        return self.data_shape
    
    @property
    def num_examples(self):
        return len(self.labels)
    
    
    def sample_latent_factors(self, num):
        """
        Sample a batch of latent factors, Y'
        Args:
            num: int, number of examples to generate
        
        Returns:
            factors: torch.Tensor, The set of latent factors
        """
        factors = torch.rand(num, len(self.observed_latent_factor_indices), generator=self.rand_generator)
        factors = (factors*torch.index_select(self.latent_sizes, 0, torch.tensor(self.observed_latent_factor_indices))).int().floor()
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
        all_factors = torch.zeros(num_samples, len(self.latent_sizes)).int()
        all_factors[:,self.observed_latent_factor_indices] = observed_latent_factors

        if len(self.unobserved_latent_factor_indices) > 0:
            rem_unobserved_latent_factors = torch.rand(num_samples, len(self.unobserved_latent_factor_indices), generator=self.rand_generator)
            all_factors[:,self.unobserved_latent_factor_indices] = (rem_unobserved_latent_factors*torch.index_select(self.latent_sizes, 0, torch.tensor(self.unobserved_latent_factor_indices))).int().floor()
        
        #Final sanity check
        assert torch.all(all_factors[:,self.observed_latent_factor_indices] == observed_latent_factors), "The observed latent factors are not correctly placed"

        return all_factors.int()
    
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
        if observed_factors is None:
            all_factors = self.sample_full_latent_vector(self.sample_latent_factors(num))
        else:
            all_factors = self.sample_full_latent_vector(observed_factors)

        images = []
        for i in range(num):
            rank = self.latent2rank[all_factors[i,0].item()]
            suit = self.latent2suit[all_factors[i,1].item()]
            label = f'{rank}{suit}'
            image_path = os.path.join(self.data_dir, self.labels[label])
            images.append(torchvision.io.read_image(image_path)[1:,:,:])

        images = torch.stack(images) if len(images) > 1 else images[0].unsqueeze(0)

        return images if return_factors else images
    

    def sample_paired_observations_from_factors(self, num=1, k = -1, observed_idx='constant', return_factors=False):
        """
        Generate the required paired examples for Weak Disentanglement
        Args:
            num: int, number of examples to generate
            k: int, number of uncommon factors betweem X1 and X2, if set to -1, it is randomly sampled between 0 and len(observed_latent_factor_indices)
            observed_idx: str, 'constant' or 'random', if 'constant', then the k-different factors are the same for all pairs, if 'random', then the k-different factors are randomly sampled for each pair
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
                k_observed = torch.randint(1, len(self.observed_latent_factor_indices), (1,), generator=self.rand_generator)
            else:
                k_observed = torch.randint(1, len(self.observed_latent_factor_indices), (num,), generator=self.rand_generator)
        else:
            k_observed = torch.tensor(k)

        if observed_idx == 'constant':
            indices = torch.randperm(len(self.observed_latent_factor_indices), generator=self.rand_generator)[:k_observed]
            indices = indices.repeat(num, 1)
        else:
            indices = torch.rand(num, len(self.observed_latent_factor_indices), generator=self.rand_generator)
            indices = torch.argsort(indices, dim=1)[:,:k_observed]

        label = indices[:,-1]

        #Since assumption is based on the fact that only a single factor is different
        #We can just randomly sample the k-different factors for each pair
        replacement_latents = self.sample_latent_factors(num)[:,indices]

        #Now we replace the k-different factors for each pair
        all_factors_x2[:,indices] = replacement_latents

        #Now we get the images for the pairs
        images_x1 = self.sample_observations(num, all_factors_x1, return_factors=False)
        images_x2 = self.sample_observations(num, all_factors_x2, return_factors=False)

        all_images = torch.cat((images_x1,images_x2), dim=2)

        return all_images, label, (all_factors_x1, all_factors_x2) if return_factors else all_images, label

