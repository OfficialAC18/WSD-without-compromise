import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from functools import partial
from utils import losses


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
        raise NotImplementedError()

    @abstractmethod
    def decoder(self, z):
        """
        Decoder function
        Args:
            z: torch.Tensor, latent sample
        Returns:
            x: torch.Tensor, reconstructed data
        """
        raise NotImplementedError()

    @abstractmethod
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick
        Args:
            mu: torch.Tensor, mean of the latent distribution
            log_var: torch.Tensor, log variance of the latent distribution
        """
        raise NotImplementedError()



class VAE(GaussianEncoderDecoderModel):
    """
    Implementation of a Variational Autoencoder. \n
    Uses a Convolutional Model as the encoder-decoder structures.
    """
    
    def __init__(self, data_shape, num_channels = 1, latent_dim=10, reconstruction_loss = 'bernoulli', beta=1):
        super().__init__()
        self.data_shape = data_shape
        self.beta = beta
        self.latent_dim = latent_dim
        self.labels = False

        if reconstruction_loss == 'bernoulli':
            self.reconstruction_loss = losses.bernoulli_loss
        elif reconstruction_loss == 'l2':
            self.reconstruction_loss = losses.l2_loss

        self.enc_conv = nn.Sequential(
            nn.Conv2d(num_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU()
        )

        # Get shape after convolutions for the decoder
        with torch.no_grad():
            dummy = torch.zeros(1, *data_shape)
            self.feat_shape = self.enc_conv(dummy).shape[1:]
            self.flat_size = torch.prod(torch.tensor(self.feat_shape)).item()

        self.enc_fc = nn.Sequential(
            nn.Linear(self.flat_size, 256),
            nn.ReLU()
        )

        self.z_mean_head = nn.Linear(256, latent_dim)
        self.z_logvar_head= nn.Linear(256, latent_dim)

        self.dec_fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.flat_size),
            nn.ReLU()
        )

        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, num_channels, 4, stride=2, padding=1)
        )
    
    def forward(self, x, labels = None):
        # We only need the first feature for the VAE
        features = x[:, :, :self.data_shape[1], :]
        z_mean, z_logvar = self.encoder(features)
        z_sampled = self.reparameterize(z_mean, z_logvar)
        x_recons = self.decoder(z_sampled)
        recon_loss = self.reconstruction_loss(features, x_recons)
        kl_loss = losses.compute_gaussian_kl(z_mean, z_logvar)
        regularizer = self.regularizer(kl_loss)
        loss = recon_loss + regularizer
        elbo = recon_loss + kl_loss
        return x_recons, x_recons, loss, elbo


    def encoder(self, x):
        h = self.enc_conv(x)
        h = torch.flatten(h, start_dim=1)
        h = self.enc_fc(h)
        return self.z_mean_head(h), self.z_logvar_head(h)

    def decoder(self, z):
        h = self.dec_fc(z)
        h = h.view(-1, *self.feat_shape)
        return self.dec_conv(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def regularizer(self, kl_loss):
        return self.beta * kl_loss



