import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from models.VAE import VAE
#Things to test:
#1. Check if the the shape of image after decoding is same as the input image. (Done)
#2. Check the shape of the latent representation. (Done)
#3. Check that the mean and variance are not the exact same. (Done)
#5. Check that the mean and variance are not None. (Done)
#6. Check that the returned mean and variance values are the same as the ones stored in the object.
mps_device = torch.device('mps')

@pytest.mark.parametrize('data_shape, latent_dim',[
                          [(3, 128, 128), 6],
                          [(3, 64, 64), 10],
                          [(1, 64, 64), 10],
                          [(3, 96, 96), 10],
                          [(1, 96, 96), 10],
                          [(3, 523, 831), 10]])

def test_forward(data_shape, latent_dim):
    #Load model
    model = VAE(data_shape=data_shape,
                latent_dim=latent_dim,
                num_channels=data_shape[0])

    #Push to device
    model = model.to(mps_device)

    #Create a random image
    x = torch.randn(64, *data_shape).to(mps_device)

    #Forward pass
    y, z_mean, z_logvar = model(x)


    #Check if the shape of the image is the same
    assert y.shape == x.shape
    #Check that None of the values are NaN
    assert not torch.isnan(y).any()
    #Check the dimensions of the latent representation
    assert model.z_mean.shape == (64,latent_dim) and model.z_logvar.shape == (64,latent_dim)
    #Check that the mean and variance are not the exact same
    assert not torch.equal(model.z_mean, model.z_logvar)
    #Check that the mean and variance are not None
    assert not torch.isnan(model.z_mean).any() and not torch.isnan(model.z_logvar).any()
    #Check that the returned mean and variance values are the same as the ones stored in the object
    assert torch.equal(z_mean, model.z_mean) and torch.equal(z_logvar, model.z_logvar)

