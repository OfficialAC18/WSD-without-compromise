import torch
import torch.nn.functional as F

def bernoulli_loss(x_true,x_recons):

    """
    Computes the Bernoulli Loss between the true image and the reconstructed image
    Args:
        x_true: torch.Tensor, true image
        x_recons: torch.Tensor, reconstructed image
    Returns:
        loss: torch.Tensor, Bernoulli loss
    """

    #Flatten the images
    x_true_reshaped = torch.reshape(x_true, (x_true.shape[0], -1))
    x_recons_reshaped = torch.reshape(x_recons, (x_recons.shape[0], -1))

    #Calculate sigmoid cross entropy
    loss = (F.binary_cross_entropy_with_logits(input=x_recons_reshaped, target=x_true_reshaped, reduction='sum') /
            x_true.size(0))
    return loss


def l2_loss(x_true,x_recons):
    """
    Computes the L2 Loss between the true image and the reconstructed image
    Args:
        x_true: torch.Tensor, true image
        x_recons: torch.Tensor, reconstructed image
    Returns:
        loss: torch.Tensor, L2 loss
    """
    x_true_reshaped = torch.reshape(x_true, (x_true.shape[0], -1))
    x_recons_reshaped = torch.reshape(torch.nn.Sigmoid()(x_recons), (x_recons.shape[0], -1))
    loss = F.mse_loss(input=x_recons_reshaped, target=x_true_reshaped, reduction='sum') / x_true.size(0)
    return loss


def compute_gaussian_kl(z_mean, z_logvar):
    """
    Compute KL divergence between input Gaussian and standard Gaussian
    Args:
        z_mean: torch.Tensor, mean of the Gaussian
        z_logvar: torch.Tensor, log variance of the Gaussian
    Returns:
        kl_loss: torch.Tensor, KL divergence
    """
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp()) / z_mean.size(0)
    return kl_loss

def compute_kl(z_mean_1, z_logvar_1, z_mean_2, z_logvar_2):
    """
    Compute KL divergence between two Gaussians
    Args:
        z_mean_1: torch.Tensor, mean of the Gaussian 1
        z_logvar_1: torch.Tensor, log variance of the Gaussian 1
        z_mean_2: torch.Tensor, mean of the Gaussian 2
        z_logvar_2: torch.Tensor, log variance of the Gaussian 2
    Returns:
        kl_loss: torch.Tensor, KL divergence
    """
    var_1 = torch.exp(z_logvar_1)
    var_2 = torch.exp(z_logvar_2)
    return var_1/var_2 + (z_mean_2 - z_mean_1)**2/var_2 - 1 + z_logvar_2 - z_logvar_1