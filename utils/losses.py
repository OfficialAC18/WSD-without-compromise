import torch
import torch.nn.functional as F

def bernoulli_loss(x_true,x_recons,
                   subtract_true_image_entropy=False):
    
    """
    Computes the Bernoulli Loss between the true image and the reconstructed image
    Args:
        x_true: torch.Tensor, true image
        x_recons: torch.Tensor, reconstructed image
        activation: torch.nn.Module, activation function
        subract_true_image_entropy: bool, whether to subtract the entropy of the true image
    Returns:
        loss: torch.Tensor, Bernoulli loss
    """

    #Flatten the images
    x_true_reshaped = torch.reshape(x_true, (x_true.shape[0], -1))
    x_recons_reshaped = torch.reshape(x_recons, (x_recons.shape[0], -1))  

    if subtract_true_image_entropy:
        dist = torch.distributions.bernoulli.Bernoulli(probs=torch.clamp(x_true_reshaped, 1e-6, 1 - 1e-6))
        loss_lower_bound = torch.sum(dist.entropy(),dim=1)
    else:
        loss_lower_bound = 0
    
    #Calculate sigmoid cross entropy
    loss = torch.sum(F.binary_cross_entropy_with_logits(input=F.sigmoid(x_recons_reshaped), target=x_true_reshaped,
                                            reduction='none'), dim = 1)
    
    return loss - loss_lower_bound


def l2_loss(x_true,x_recons):
    """
    Computes the L2 Loss between the true image and the reconstructed image
    Args:
        x_true: torch.Tensor, true image
        x_recons: torch.Tensor, reconstructed image
    Returns:
        loss: torch.Tensor, L2 loss
    """
    return torch.sum((x_true - torch.nn.Sigmoid()(x_recons))**2,dim=1)


def compute_gaussian_kl(z_mean, z_logvar):
    """
    Compute KL diversgence between input Gaussian and standard Gaussian
    Args:
        z_mean: torch.Tensor, mean of the Gaussian
        z_logvar: torch.Tensor, log variance of the Gaussian
    Returns:
        kl_loss: torch.Tensor, KL divergence
    """
    kl_loss = 0.5 * torch.mean(torch.sum(z_mean**2 + torch.exp(z_logvar) - z_logvar - 1, dim=1))
    return kl_loss

def compute_kl(z_mean_1, z_logvar_1, z_mean_2, z_logvar_2):
    """
    Compute KL diversgence between two Gaussians
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