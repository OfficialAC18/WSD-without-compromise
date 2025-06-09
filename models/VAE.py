import torch
import torch.nn.functional as F
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
    
    def __init__(self, data_shape, num_channels = 1, latent_dim=10):
        super().__init__()
        self.z_mean = None
        self.z_logvar = None
        self.data_shape = data_shape

        #Defining the encoder components
        self.enc_1 = torch.nn.Conv2d(in_channels=num_channels,
                                     out_channels=32,
                                     kernel_size=4,
                                     stride=2)
        
        self.enc_2 = torch.nn.Conv2d(in_channels=32,
                                     out_channels=32,
                                     kernel_size=4,
                                     stride=2)
        
        self.enc_3 = torch.nn.Conv2d(in_channels=32,
                                     out_channels=64,
                                     kernel_size=4,
                                     stride=2)
        
        self.enc_4 = torch.nn.Conv2d(in_channels=64,
                                     out_channels=64,
                                     kernel_size=4,
                                     stride=2)
        
        self.enc_5 = torch.nn.LazyLinear(out_features=256)

        self.z_mean_head = torch.nn.Linear(
            in_features=256,
            out_features=latent_dim
        )     

        self.z_logvar_head = torch.nn.Linear(
            in_features=256,
            out_features=latent_dim
        )

        #Pass throught the encoder components to initlalize the lazy linear layer
        self.enc_1_output_shape = self.enc_1(torch.randn(1, *data_shape)).shape
        self.enc_2_output_shape = self.enc_2(self.enc_1(torch.randn(1, *data_shape))).shape
        self.enc_3_output_shape = self.enc_3(self.enc_2(self.enc_1(torch.randn(1, *data_shape)))).shape
        self.enc_4_output_shape = self.enc_4(self.enc_3(self.enc_2(self.enc_1(torch.randn(1, *data_shape))))).shape
        self.enc_5(torch.flatten(self.enc_4(self.enc_3(self.enc_2(self.enc_1(torch.randn(1, *data_shape)))))).reshape(1,-1))
        

        #Defining the decoder components
        self.dec_1 = torch.nn.Linear(in_features=latent_dim,
                                     out_features=256)
        
        self.dec_2 = torch.nn.Linear(in_features=256,
                                     out_features=self.enc_5.weight.shape[1])
        
        self.dec_3 = torch.nn.ConvTranspose2d(in_channels=64,
                                              out_channels=64,
                                              kernel_size=4,
                                              stride=2,
                                              )
        
        self.dec_4 = torch.nn.ConvTranspose2d(in_channels=64,
                                              out_channels=32,
                                              kernel_size=4,
                                              stride=2)
        
        self.dec_5 = torch.nn.ConvTranspose2d(in_channels=32,
                                            out_channels=32,
                                            kernel_size=4,
                                            stride=2)
        
        self.dec_6 = torch.nn.ConvTranspose2d(in_channels=32,
                                            out_channels=num_channels,
                                            kernel_size=4,
                                            stride=2)
    
    def forward(self, x):
        z_mean, z_logvar = self.encoder(x)
        z = self.reparameterize(z_mean, z_logvar)
        x_recons = self.decoder(z)
        return x_recons, z_mean, z_logvar


    def encoder(self, x):
        x = F.relu(self.enc_1(x))
        x = F.relu(self.enc_2(x))
        x = F.relu(self.enc_3(x))
        x = F.relu(self.enc_4(x))

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.enc_5(x))

        #Get Mean and Log Variance
        z_mean = self.z_mean_head(x)
        z_logvar = self.z_logvar_head(x)

        self.z_mean = z_mean
        self.z_logvar = z_logvar

        return z_mean, z_logvar

    def decoder(self, z):
        z = self.dec_1(z)
        z = self.dec_2(z)
        z = self.dec_3(torch.reshape(z, (-1, *self.enc_4_output_shape[1:])),
                       output_size=self.enc_3_output_shape)
        z = self.dec_4(z,output_size=self.enc_2_output_shape)
        z = self.dec_5(z,output_size=self.enc_1_output_shape)
        output = self.dec_6(z,output_size=(-1,*self.data_shape))
        # output = torch.reshape(output, (-1, *self.data_shape))
        return output

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std



