import torch
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

    @abstractmethod
    def sample(self):
        """
        Sample from the model
        Args:
        Returns:
            x: torch.Tensor, generated sample
        """
        return torch.normal(mean=self.z_mean,
                            std=self.z_logstd)


class VAE(GaussianEncoderDecoderModel):
    """
    Implementation of a Variational Autoencoder
    Uses a Convolutional Model as the encoder-decoder structures
    """
    
    def __init__(self, output_shape, num_channels = 1, latent_dim=10):
        self.z_mean = None
        self.z_logvar = None
        self.z_logmean = None
        self.output_shape = output_shape

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
        




    def encoder(self, x):
        x = self.enc_1(x)
        x = self.enc_2(x)
        x = self.enc_3(x)
        x = self.enc_4(x)
        x = self.enc_5(x)

        #Get Mean and Log Variance
        self.z_mean = self.z_mean_head(x)
        self.z_logvar = self.z_logvar_head(x)

    def decoder(self, z):
        z = self.dec_1(z)
        z = self.dec_2(z)
        z = self.dec_3(torch.reshape(z, (-1, 64, 4, 4)))
        z = self.dec_4(z)
        z = self.dec_5(z)
        output = self.dec_6(z)
        output = torch.reshape(output, (-1, self.output_shape))
        return output

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def generate(self):
        return self.decoder(self.reparameterize(self.z_mean, self.z_logvar))
