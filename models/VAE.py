import torch
import torch.nn.functional as F

from utils import losses
from functools import partial
from abc import ABC, abstractmethod



class GaussianEncoderDecoderModel(torch.nn.Module, ABC):
    """
    Abstract Class for Gaussian Encoder-Decoder Models (VAEs basically)
    """

    @abstractmethod
    def encoder(self, x):
        """
        Encoder function
        Args:
            x: torch.Tensor, input data
        Returns:
            mu: torch.Tensor, mean of the latent distribution
            log_var: torch.Tensor, log variance of the latent distribution
        """
        pass

    @abstractmethod
    def decoder(self, z):
        """
        Decoder function
        Args:
            z: torch.Tensor, latent sample
        Returns:
            x: torch.Tensor, reconstructed data
        """
        pass



class VAE(GaussianEncoderDecoderModel):
    """
    Implementation of a Variational Autoencoder
    Uses a Convolutional Model as the encoder-decoder structures
    """
    
    def __init__(self, data_shape, num_channels = 1, latent_dim=10):
        super().__init__()
        self.z_mean = None
        self.z_logvar = None
        self.z_logmean = None
        self.data_shape = data_shape

        #Defining the encoder components
        self.enc_1 = torch.nn.Conv2d(in_channels=1,
                                     out_channels=32,
                                     kernel_size=4,
                                     stride=2,
                                     padding='same')
        
        self.enc_2 = torch.nn.Conv2d(in_channels=32,
                                     out_channels=32,
                                     kernel_size=4,
                                     stride=2,
                                     padding='same')
        
        self.enc_3 = torch.nn.Conv2d(in_channels=32,
                                     out_channels=64,
                                     kernel_size=4,
                                     stride=2,
                                     padding='same')
        
        self.enc_4 = torch.nn.Conv2d(in_channels=64,
                                     out_channels=64,
                                     kernel_size=4,
                                     stride=2,
                                     padding='same')
        
        self.enc_5 = torch.nn.Linear(in_features=64*6*2,
                                     out_features=256)

        self.z_mean_head = torch.nn.Linear(
            in_features=256,
            out_features=latent_dim
        )     

        self.z_logvar_head = torch.nn.Linear(
            in_features=256,
            out_features=latent_dim
        )


        #Defining the decoder components
        self.dec_1 = torch.nn.Linear(in_features=latent_dim,
                                     out_features=256)
        
        self.dec_2 = torch.nn.Linear(in_features=256,
                                     out_features=1024)
        
        self.dec_3 = torch.nn.ConvTranspose2d(in_channels=64,
                                              out_channels=64,
                                              kernel_size=4,
                                              stride=2,
                                              padding='same')
        
        self.dec_4 = torch.nn.ConvTranspose2d(in_channels=64,
                                              out_channels=32,
                                              kernel_size=4,
                                              stride=2,
                                              padding='same')
        
        self.dec_5 = torch.nn.ConvTranspose2d(in_channels=32,
                                            out_channels=32,
                                            kernel_size=4,
                                            stride=2,
                                            padding='same')
        
        self.dec_5 = torch.nn.ConvTranspose2d(in_channels=32,
                                            out_channels=num_channels,
                                            kernel_size=4,
                                            stride=2,
                                            padding='same')
    
    def forward(self, x):
        z = self.encoder(x)
        x_recons = self.decoder(z)
        return x_recons


    def encoder(self, x):
        x = F.relu(self.enc_1(x))
        x = F.relu(self.enc_2(x))
        x = F.relu(self.enc_3(x))
        x = F.relu(self.enc_4(x))
        x = F.relu(self.enc_5(x))

        #Get Mean and Log Variance
        self.z_mean = self.z_mean_head(x)
        self.z_logvar = self.z_logvar_head(x)

    def decoder(self, z):
        z = self.dec_1(z)
        z = self.dec_2(z)
        z = self.dec_3(torch.reshape(z, (-1, 64, 4, 4)))
        z = self.dec_4(z)
        output = self.dec_5(z)
        output = torch.reshape(output, (-1, self.data_shape))
        return output

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std



class GroupVAEBase(VAE):
    """
    Beta-VAE with averaging from https://arxiv.org/abs/1809.02383.
    with additional averaging for weak supervision
    Args:
        output_shape: tuple, shape of the output data
        num_channels: int, number of channels in the output data
        latent_dim: int, dimension of the latent space
        beta: float, beta parameter for KL divergence
    """
    def __init__(self, output_shape, num_channels = 1, labels = None,
                latent_dim=10, beta = 1.0, reconstruction_loss = 'bernoulli',subtract_true_image_entropy = False):
        super().__init__(output_shape, num_channels, latent_dim)
        self.beta = beta
        self.labels = labels
        if reconstruction_loss == 'bernoulli':
            self.reconstruction_loss = partial(losses.bernoulli_loss,
                                               subtract_true_image_entropy=subtract_true_image_entropy)
        elif reconstruction_loss == 'l2':
            self.reconstruction_loss = losses.l2_loss

    def regularizer(self, kl_loss):
        return self.beta * kl_loss 
    
    def aggregate(self, z_mean_1, z_logvar_1, z_mean_2, z_logvar_2):
        pass
    
    def forward(self, x):
        features_x1 = x[:, :, :self.data_shape[1], :]
        features_x2 = x[:, :, self.data_shape[1]:, :]

        #Get the latent representations of both the examples
        z_mean_1, z_logvar_1 = self.encoder(features_x1)
        z_mean_2, z_logvar_2 = self.encoder(features_x2)  

        #Point-wise KL divergence
        per_point_kl = losses.compute_kl(z_mean_1, z_logvar_1, z_mean_2, z_logvar_2)

        #Calculate the average representation
        z_mean_avg = 0.5*(z_mean_1 + z_mean_2)
        z_logvar_avg = 0.5*(torch.exp(z_logvar_1) + torch.exp(z_logvar_2))

        #Aggregate the representations
        z_agg_1, z_agg_logvar_1 = self.aggregate(z_mean_1, z_logvar_1,
                                            z_mean_avg, z_logvar_avg,
                                            per_point_kl)
        
        z_agg_2, z_agg_logvar_2 = self.aggregate(z_mean_2, z_logvar_2,
                                            z_mean_avg, z_logvar_avg,
                                            per_point_kl)
        
        #Sample using distributions
        z_sampled_1 = self.reparameterize(z_agg_1, z_agg_logvar_1)
        z_sampled_2 = self.reparameterize(z_agg_2, z_agg_logvar_2)

        #Reconstruct the images
        x_recons_1 = self.decoder(z_sampled_1)
        x_recons_2 = self.decoder(z_sampled_2)

        #Calculate the reconstruction loss
        reconstruction_loss_1 = torch.mean(self.reconstruction_loss(features_x1, x_recons_1))
        reconstruction_loss_2 = torch.mean(self.reconstruction_loss(features_x2, x_recons_2))
        reconstruction_loss = 0.5*(reconstruction_loss_1 + reconstruction_loss_2)

        #Calculate the KL divergence
        kl_loss_1 = losses.compute_gaussian_kl(z_mean_1, z_logvar_1)
        kl_loss_2 = losses.compute_gaussian_kl(z_mean_2, z_logvar_2)
        kl_loss = 0.5*(kl_loss_1 + kl_loss_2)

        #Regularizing KL Loss
        regularizer = self.regularizer(kl_loss)

        loss = reconstruction_loss + regularizer
        elbo = reconstruction_loss + kl_loss

        return x_recons_1, x_recons_2, loss, -elbo