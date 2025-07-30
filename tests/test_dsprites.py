import pytest
import torch
import os
import sys
import unittest.mock as mock

# Add the project root to the Python path to allow importing 'data' module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dsprites import Dsprites

# Define the path to the dataset (adjust if necessary)
# Assuming the script is run from the project root or tests directory
DATA_DIR = 'datasets/dSprites'

@pytest.fixture(scope="module")
def data_dir():
    """Fixture for the dSprites data directory."""
    dir_path = os.path.abspath(DATA_DIR)
    if not os.path.exists(os.path.join(dir_path, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')):
        pytest.skip(f"dSprites dataset not found in {dir_path}. Skipping tests.")
    return dir_path

# --- Fixtures for Dsprites instances with different observed factors ---

@pytest.fixture(scope="module")
def dsprites_all_observed(data_dir):
    """Dsprites instance with all factors observed."""
    return Dsprites(data_dir=data_dir, observed_latent_factor_indices=None, seed=42)

@pytest.fixture(scope="module")
def dsprites_some_observed(data_dir):
    """Dsprites instance with specific factors observed."""
    return Dsprites(data_dir=data_dir, observed_latent_factor_indices=[1, 3, 4], seed=42) # Shape, Orientation, PosX

@pytest.fixture(scope="module")
def dsprites_one_observed(data_dir):
    """Dsprites instance with only one factor observed."""
    return Dsprites(data_dir=data_dir, observed_latent_factor_indices=[2], seed=42) # Scale


# Combine fixtures for parameterization
dsprites_instances = ["dsprites_all_observed", "dsprites_some_observed", "dsprites_one_observed"]

# --- Test __init__ and basic properties ---

@pytest.mark.parametrize("instance_name, expected_observed, expected_unobserved", [
    ("dsprites_all_observed", [0, 1, 2, 3, 4, 5], []),
    ("dsprites_some_observed", [1, 3, 4], [0, 2, 5]),
    ("dsprites_one_observed", [2], [0, 1, 3, 4, 5]),
])
def test_init(instance_name, expected_observed, expected_unobserved, request):
    """Test that all values are set correctly during init"""
    dsprites = request.getfixturevalue(instance_name)

    assert dsprites.data is not None
    assert isinstance(dsprites.images, torch.Tensor)
    assert dsprites.images.shape[0] == 737280  # Total number of images
    assert dsprites.images.shape[1:] == (64, 64)
    assert dsprites.data_shape == (1,64,64)
    assert isinstance(dsprites.latent_factor_sizes, torch.Tensor)
    assert torch.equal(dsprites.latent_factor_sizes, torch.tensor([1, 3, 6, 40, 32, 32]))
    assert isinstance(dsprites.factor_bases, torch.Tensor)
    assert dsprites.factor_bases.dtype == torch.float32  # Result of division
    assert dsprites.observed_latent_factor_indices == expected_observed
    assert dsprites.unobserved_latent_factor_indices == expected_unobserved
    assert dsprites.rand_generator is not None
    assert isinstance(dsprites.rand_generator, torch.Generator)


@pytest.mark.parametrize("instance_name", dsprites_instances)
def test_properties(instance_name, request):
    """Test the property methods of the Dsprites class."""
    dsprites = request.getfixturevalue(instance_name)
    expected_num_observed = len(dsprites.observed_latent_factor_indices)
    assert dsprites.num_observed_latent_factors == expected_num_observed

    if expected_num_observed > 0:
        expected_latent_sizes = [dsprites.latent_factor_sizes[i] for i in dsprites.observed_latent_factor_indices]
        assert [s.item() for s in dsprites.observed_latent_factor_sizes] == [s.item() for s in expected_latent_sizes]
    else:
        assert dsprites.observed_latent_factor_sizes == []

    assert dsprites.example_shape == (1, 64, 64)
    assert dsprites.num_examples == 737280

# --- Test sampling methods ---

@pytest.mark.parametrize("instance_name", dsprites_instances)
@pytest.mark.parametrize("num_samples", [1, 5, 10])
def test_sample_latent_factors(instance_name, num_samples, request):
    """Test sampling of observed latent factors."""
    dsprites = request.getfixturevalue(instance_name)
    num_observed = dsprites.num_observed_latent_factors

    factors = dsprites.sample_latent_factors(num_samples)
    assert factors.shape == (num_samples, 6)
    assert factors.dtype == torch.int32

    # Check if values are within the allowed range for each observed factor
    assert torch.all(factors >= 0)
    assert torch.all(factors < dsprites.latent_factor_sizes)

@pytest.mark.parametrize("instance_name", dsprites_instances)
@pytest.mark.parametrize("num_samples", [1, 3])
@pytest.mark.parametrize("return_factors", [True, False])
def test_sample_observations(instance_name, num_samples, return_factors, request):
    """Test sampling observations without providing factors."""
    dsprites = request.getfixturevalue(instance_name)
    total_factors = len(dsprites.latent_factor_sizes)
    img_shape = dsprites.example_shape

    # Hard-code the factors that sample_latent_factors returns
    latent_factors = torch.tensor([[0, 0, 0, 38, 12, 27], [0, 0, 2, 12, 1, 27], [0, 0, 2, 29, 11, 29]])
    with mock.patch.object(dsprites, 'sample_latent_factors', return_value = latent_factors[:num_samples]):
        result = dsprites.sample_observations(num_samples, return_factors=return_factors)

    if return_factors:
        images, factors = result
        assert isinstance(factors, torch.Tensor)
        assert factors.shape == (num_samples, total_factors)
    else:
        images = result
    assert isinstance(images, torch.Tensor)
    assert images.shape == (num_samples, *img_shape[1:]) # without channel dim
    assert images.dtype == torch.uint8

    # Test that the returned images are correct given the factors
    # [0, 0, 0, 38, 12, 27] is entry 27+12*32+38*32*32=39323
    # [0, 0, 2, 12, 1, 27] is entry 27+32+12*32*32+2*32*32*40=94267
    # [0, 0, 2, 29, 11, 29] is entry 29+11*32+29*32*32+2*32*32*40=111997
    correct_images = dsprites.images[[39323,94267,111997]]
    assert torch.all(images == correct_images[:num_samples])


# --- Test paired sampling ---

@pytest.mark.parametrize("instance_name", ["dsprites_all_observed", "dsprites_some_observed"]) # Need > 0 observed factors
@pytest.mark.parametrize("num", [1, 5, 10])
@pytest.mark.parametrize("k", [-1, 0, 1, 2])
@pytest.mark.parametrize("observed_idx_mode", ['constant', 'random'])
@pytest.mark.parametrize("return_factors", [True, False])
def test_sample_paired_observations_from_factors(instance_name, num, k, observed_idx_mode, return_factors, request):
    """Test sampling paired observations with varying common factors."""
    dsprites = request.getfixturevalue(instance_name)
    num_observed = dsprites.num_observed_latent_factors
    total_factors = len(dsprites.latent_factor_sizes)
    img_shape = dsprites.example_shape # (1, 64, 64)
    C, H, W = img_shape

    if k > num_observed:
        pytest.skip(f"Skipping test: k={k} is greater than num_observed={num_observed}")
    if k == -1 and num_observed == 0:
         pytest.skip(f"Skipping test: k={k} requires num_observed > 0")
    if k >= 0 and observed_idx_mode == 'random':
        # observed_idx='random' only makes sense when k=-1 (to determine k per sample)
        # If k is fixed, observed_idx mode doesn't matter for k selection.
        pytest.skip("Skipping test: observed_idx='random' only relevant for k=-1")


    result = dsprites.sample_paired_observations(
        num_samples=num,
        k=k,
        observed_idx=observed_idx_mode,
        return_latents=return_factors
    )

    if return_factors:
        image_pairs, labels, factors_x1, factors_x2 = result
        assert isinstance(factors_x1, torch.Tensor)
        assert factors_x1.shape == (num, total_factors)
        assert isinstance(factors_x2, torch.Tensor)
        assert factors_x2.shape == (num, total_factors)

        obs_indices = dsprites.observed_latent_factor_indices

        for i in range(num):
            # Observed factors should match except for the 'k' differing ones (which might still match by chance)
            match_observed_factors_count = 0

            for factor_idx in obs_indices:
                if factors_x1[i, factor_idx] == factors_x2[i, factor_idx]:
                    match_observed_factors_count += 1

            # Verify number of differing factors based on k
            if k == -1:
                # at least one factor should stay the same
                assert match_observed_factors_count > 0
            else:
                # at least all unchanged factors should stay the same
                assert match_observed_factors_count >= num_observed - k
    else:
        image_pairs, labels = result

    assert isinstance(image_pairs, torch.Tensor)
    assert image_pairs.shape == (num, C, 2*H, W)
    assert image_pairs.dtype == torch.float32

    assert labels.shape == (num, total_factors)
    # Check that k entries are marked in label vector
    if k == -1:
        assert 0 < labels.sum() < num_observed * num
    else:
        assert labels.sum() == k * num
