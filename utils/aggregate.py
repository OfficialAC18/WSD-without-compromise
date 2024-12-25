import torch

def aggregate_labels(z_mean,z_logvar,
                    new_mean, new_logvar, labels):
    """
    Aggregation of representations using labels
    
    Labels are one-hot encoded vectors specifying the latent factors that are NOT shared.
    Function enforces that each factor of variation is encoded in single 
    dimesnion in latent representation. Function also enforces predicatable mapping
    for each factor of variation in latent dimension (factor 1 -> dim 1, factor 2 -> dim 2 etc.).

    Args:
        z_mean: torch.Tensor, mean of encoder distrbution of original image
        z_logvar: torch.Tensor, log variance of encoder distribution of original image
        new_mean: torch.Tensor, mean of encoder distributions of pair of images
        new_logvar: torch.Tensor, log variance of encoder distributions of pair of images
        labels: torch.Tensor, one-hot encoded labels

    Returns:
        z_agg: torch.Tensor, aggregated mean
        z_agg_logvar: torch.Tensor, aggregated log variance
    """

    z_agg = torch.where(labels == 1, z_mean, new_mean)
    z_agg_logvar = torch.where(labels == 1, z_logvar, new_logvar)
    return z_agg, z_agg_logvar


def aggregate_max(z_mean, z_logvar,
                  new_mean, new_logvar, per_point_kl):
    """
    Aggregation of representations using maximum KL divergence

    The dimensions with the minimum KL divergence are not aggregated. We aggregate
    the max K dimensions to be aggragated, we adaptively estimate K by using the thereshold.

                                τ=1/2(max δi + min δi)

    Args:
        z_mean: torch.Tensor, mean of encoder distrbution of original image
        z_logvar: torch.Tensor, log variance of encoder distribution of original image
        new_mean: torch.Tensor, mean of encoder distributions of pair of images
        new_logvar: torch.Tensor, log variance of encoder distributions of pair of images
        per_point_kl: torch.Tensor, KL divergence between the encoder distributions

    Returns:
        z_agg: torch.Tensor, aggregated mean
        z_agg_logvar: torch.Tensor, aggregated log variance
    """
    thereshold = 0.5 * (torch.max(per_point_kl) + torch.min(per_point_kl))
    z_agg = torch.where(per_point_kl > thereshold, z_mean, new_mean)
    z_agg_logvar = torch.where(per_point_kl > thereshold, z_logvar, new_logvar)
    return z_agg, z_agg_logvar