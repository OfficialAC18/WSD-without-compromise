import torch

from utils import losses
from functools import partial

from VAE import VAE


class MLVAEBase(VAE):
    """
    Beta-VAE with averaging from https://arxiv.org/abs/1809.02383.
    with additional averaging for weak supervision
    Args:
        data_shape: tuple, shape of the output data
        num_channels: int, number of channels in the output data
        labels: torch.Tensor, labels for weak supervision (Optional)
        latent_dim: int, dimension of the latent space
        beta: float, beta parameter for KL divergence
        reconstruction_loss: str, type of reconstruction loss (bernoulli or l2)
        subtract_true_image_entropy: bool, whether to subtract the entropy of the true image (in case of bernoulli loss)
    """
    def __init__(self, data_shape, num_channels = 1, labels = None,
                latent_dim=10, beta = 1.0, reconstruction_loss = 'bernoulli',subtract_true_image_entropy = False):
        super().__init__(data_shape, num_channels, latent_dim)
        self.beta = beta
        self.labels = labels
        if reconstruction_loss == 'bernoulli':
            self.reconstruction_loss = partial(losses.bernoulli_loss,
                                               subtract_true_image_entropy=subtract_true_image_entropy)
        elif reconstruction_loss == 'l2':
            self.reconstruction_loss = losses.l2_loss

    def regularizer(self, kl_loss):
        return self.beta * kl_loss 
    
    def aggregate(self, z_mean_1, z_logvar_1, z_mean_avg, z_logvar_avg, per_point_kl):
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
        var_1 = torch.exp(z_logvar_1)
        var_2 = torch.exp(z_logvar_2)
        new_var = 2*var_1*var_2/(var_1 + var_2)

        z_logvar_avg = torch.log(new_var)
        z_mean_avg = (z_mean_1/var_1 + z_mean_2/var_2)*new_var*0.5
        

        #Aggregate the representations
        z_aggr_1, z_aggr_logvar_1 = self.aggregate(z_mean_1, z_logvar_1,
                                            z_mean_avg, z_logvar_avg,
                                            per_point_kl)
        
        z_aggr_2, z_aggr_logvar_2 = self.aggregate(z_mean_2, z_logvar_2,
                                            z_mean_avg, z_logvar_avg,
                                            per_point_kl)

        #Sample using distributions
        z_sampled_1 = self.reparameterize(z_aggr_1, z_aggr_logvar_1)
        z_sampled_2 = self.reparameterize(z_aggr_2, z_aggr_logvar_2)

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

