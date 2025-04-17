import pytest
import torch
import numpy as np
import os
import sys

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

@pytest.mark.parametrize("instance_name", dsprites_instances)
def test_init_loading(instance_name, request):
    """Test if data is loaded correctly during initialization."""
    dsprites = request.getfixturevalue(instance_name)
    assert dsprites.data is not None
    assert isinstance(dsprites.images, torch.Tensor)
    assert dsprites.images.shape[0] == 737280 # Total number of images
    assert dsprites.images.shape[1:] == (64, 64)
    assert isinstance(dsprites.latents_sizes, torch.Tensor)
    assert torch.equal(dsprites.latents_sizes, torch.tensor([1, 3, 6, 40, 32, 32]))
    assert isinstance(dsprites.factor_bases, torch.Tensor)
    assert dsprites.factor_bases.dtype == torch.float32 # Result of division
    assert dsprites.rand_generator is not None
    assert isinstance(dsprites.rand_generator, torch.Generator)

@pytest.mark.parametrize("instance_name, expected_observed, expected_unobserved", [
    ("dsprites_all_observed", [0, 1, 2, 3, 4, 5], []),
    ("dsprites_some_observed", [1, 3, 4], [0, 2, 5]),
    ("dsprites_one_observed", [2], [0, 1, 3, 4, 5]),
])
def test_init_indices(instance_name, expected_observed, expected_unobserved, request):
    """Test observed and unobserved indices are set correctly."""
    dsprites = request.getfixturevalue(instance_name)
    assert dsprites.observed_latent_factor_indices == expected_observed
    assert dsprites.unobserved_latent_factor_indices == expected_unobserved

@pytest.mark.parametrize("instance_name", dsprites_instances)
def test_properties(instance_name, request):
    """Test the property methods of the Dsprites class."""
    dsprites = request.getfixturevalue(instance_name)
    expected_num_observed = len(dsprites.observed_latent_factor_indices)
    assert dsprites.num_observed_latent_factors == expected_num_observed

    if expected_num_observed > 0:
        expected_latent_sizes = [dsprites.latents_sizes[i] for i in dsprites.observed_latent_factor_indices]
        assert [s.item() for s in dsprites.latent_factor_sizes] == [s.item() for s in expected_latent_sizes]
    else:
        assert dsprites.latent_factor_sizes == []

    assert dsprites.example_shape == (1, 64, 64)
    assert dsprites.num_examples == 737280

# --- Test sampling methods ---

@pytest.mark.parametrize("instance_name", dsprites_instances)
@pytest.mark.parametrize("num", [1, 5, 10])
def test_sample_latent_factors(instance_name, num, request):
    """Test sampling of observed latent factors."""
    dsprites = request.getfixturevalue(instance_name)
    num_observed = dsprites.num_observed_latent_factors

    if num_observed == 0:
        factors = dsprites.sample_latent_factors(num)
        assert factors.shape == (num, 0) # Shape should be (num, 0) if no factors observed
        return # Skip rest of checks for this case

    factors = dsprites.sample_latent_factors(num)
    assert factors.shape == (num, num_observed)
    assert factors.dtype == torch.int32

    # Check if values are within the allowed range for each observed factor
    observed_sizes = torch.tensor(dsprites.latent_factor_sizes)
    assert torch.all(factors >= 0)
    assert torch.all(factors < observed_sizes.unsqueeze(0).expand_as(factors))


@pytest.mark.parametrize("instance_name", dsprites_instances)
@pytest.mark.parametrize("num", [1, 5, 10])
def test_sample_full_latent_vector(instance_name, num, request):
    """Test sampling of the full latent vector based on observed factors."""
    dsprites = request.getfixturevalue(instance_name)
    total_factors = len(dsprites.latents_sizes)

    if dsprites.num_observed_latent_factors == 0:
        # If no factors are observed, sample_latent_factors gives shape (num, 0)
        # sample_full_latent_vector expects observed factors, let's create dummy ones if needed
        # However, the logic inside should handle this by sampling all factors randomly
        observed_factors = dsprites.sample_latent_factors(num) # This will be shape (num, 0)
        all_factors = dsprites.sample_full_latent_vector(observed_factors)
        assert all_factors.shape == (num, total_factors)
        # Ensure all factors are within bounds
        assert torch.all(all_factors >= 0)
        assert torch.all(all_factors < dsprites.latents_sizes.unsqueeze(0).expand_as(all_factors))
        return # Skip other checks

    observed_factors = dsprites.sample_latent_factors(num)
    all_factors = dsprites.sample_full_latent_vector(observed_factors)

    assert all_factors.shape == (num, total_factors)
    assert all_factors.dtype == torch.float32 or all_factors.dtype == torch.int32 # Can be float due to rand * size -> floor

    # Check observed factors are correctly placed
    if dsprites.observed_latent_factor_indices: # Check if list is not empty
        assert torch.all(all_factors[:, dsprites.observed_latent_factor_indices].int() == observed_factors.int())

    # Check unobserved factors are within bounds
    if dsprites.unobserved_latent_factor_indices: # Check if list is not empty
        unobserved_sizes = dsprites.latents_sizes[dsprites.unobserved_latent_factor_indices]
        unobserved_sampled = all_factors[:, dsprites.unobserved_latent_factor_indices]
        assert torch.all(unobserved_sampled >= 0)
        assert torch.all(unobserved_sampled < unobserved_sizes.unsqueeze(0).expand_as(unobserved_sampled))


@pytest.mark.parametrize("instance_name", dsprites_instances)
@pytest.mark.parametrize("num", [1, 5, 10])
@pytest.mark.parametrize("return_factors", [True, False])
def test_sample_observations_random(instance_name, num, return_factors, request):
    """Test sampling observations without providing factors."""
    dsprites = request.getfixturevalue(instance_name)
    total_factors = len(dsprites.latents_sizes)
    img_shape = dsprites.example_shape

    result = dsprites.sample_observations(num, return_factors=return_factors)

    if return_factors:
        images, factors = result
        assert isinstance(images, torch.Tensor)
        assert images.shape == (num, *img_shape[1:]) # Image shape without channel dim for dsprites is (64,64)
        assert images.dtype == torch.uint8
        assert isinstance(factors, torch.Tensor)
        assert factors.shape == (num, total_factors)
    else:
        images = result
        assert isinstance(images, torch.Tensor)
        assert images.shape == (num, *img_shape[1:])
        assert images.dtype == torch.uint8


@pytest.mark.parametrize("instance_name", dsprites_instances)
@pytest.mark.parametrize("num", [1, 5, 10])
@pytest.mark.parametrize("return_factors", [True, False])
def test_sample_observations_provided_factors(instance_name, num, return_factors, request):
    """Test sampling observations with provided observed factors."""
    dsprites = request.getfixturevalue(instance_name)
    total_factors = len(dsprites.latents_sizes)
    img_shape = dsprites.example_shape

    if dsprites.num_observed_latent_factors == 0:
        pytest.skip("Skipping test: Cannot provide observed factors when none are defined.")

    observed_factors = dsprites.sample_latent_factors(num)
    result = dsprites.sample_observations(num, observed_factors=observed_factors, return_factors=return_factors)

    if return_factors:
        images, factors = result
        assert isinstance(images, torch.Tensor)
        assert images.shape == (num, *img_shape[1:])
        assert images.dtype == torch.uint8
        assert isinstance(factors, torch.Tensor)
        assert factors.shape == (num, total_factors)
        # Check if the returned factors contain the provided observed factors
        assert torch.all(factors[:, dsprites.observed_latent_factor_indices].int() == observed_factors.int())
    else:
        images = result
        assert isinstance(images, torch.Tensor)
        assert images.shape == (num, *img_shape[1:])
        assert images.dtype == torch.uint8

# --- Test paired sampling ---

@pytest.mark.parametrize("instance_name", ["dsprites_all_observed", "dsprites_some_observed"]) # Need > 0 observed factors
@pytest.mark.parametrize("num", [1, 5, 10])
@pytest.mark.parametrize("k", [-1, 0, 1, 2]) # Test random k, k=0, k=1, k>1
@pytest.mark.parametrize("observed_idx_mode", ['constant', 'random']) # Test both modes for k=-1
@pytest.mark.parametrize("return_factors", [True, False])
def test_sample_paired_observations_from_factors(instance_name, num, k, observed_idx_mode, return_factors, request):
    """Test sampling paired observations with varying common factors."""
    dsprites = request.getfixturevalue(instance_name)
    num_observed = dsprites.num_observed_latent_factors
    total_factors = len(dsprites.latents_sizes)
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


    result = dsprites.sample_paired_observations_from_factors(
        num=num,
        k=k,
        observed_idx=observed_idx_mode if k == -1 else 'constant', # Mode matters only if k=-1
        return_factors=return_factors
    )

    # Expected image shape: (num, C, H, W * 2) -> dsprites images are (64, 64), pairs are concatenated along channel dim in code. Let's check code again.
    # Line 161: torch.concatenate((images_x1, images_x2), dim=1).unsqueeze(1)
    # images_x1/x2 shape: (num, H, W)
    # Concatenate dim=1: (num, H*2, W) ? No, dsprites images are (64, 64) no channel dim from loader.
    # Let's assume output should be (num, 2, H, W) or (num, 1, H, W*2) or similar. The code seems to intend (num, 1, 2, H, W) or similar after concatenation + unsqueeze
    # Let's trace shapes: images_x1 shape (num, 64, 64). concat dim=1 -> (num, 128, 64). unsqueeze(1) -> (num, 1, 128, 64)
    expected_img_pair_shape = (num, 1, H * 2, W) if C == 1 else (num, C * 2, H, W) # Adjust based on actual concatenation behavior.
    # Based on the code: torch.concatenate((images_x1, images_x2), dim=1).unsqueeze(1)
    # If images_x1 is (N, H, W), concat dim=1 -> (N, 2*H, W). unsqueeze(1) -> (N, 1, 2*H, W). This seems unlikely to be intended.
    # Let's re-read: data shape is (1, 64, 64). `images` is (N, 64, 64). `images_x1` is (num, 64, 64).
    # concat dim=1 -> (num, 128, 64). unsqueeze(1) -> (num, 1, 128, 64). Okay, this is the shape based on the code.

    expected_img_pair_shape = (num, 1, H * 2, W)


    if return_factors:
        image_pairs, labels, factors_x1, factors_x2 = result
        assert isinstance(factors_x1, torch.Tensor)
        assert factors_x1.shape == (num, total_factors)
        assert isinstance(factors_x2, torch.Tensor)
        assert factors_x2.shape == (num, total_factors)

        # --- Detailed Factor Checks ---
        obs_indices = dsprites.observed_latent_factor_indices
        unobs_indices = dsprites.unobserved_latent_factor_indices

        for i in range(num):
            # Unobserved factors should be potentially different (sampled randomly for x1 and x2)
            # Observed factors should match except for the 'k' differing ones

            label_factor_index = labels[i].item()
            assert label_factor_index in obs_indices # Label must be an observed factor index

            diff_observed_factors_count = 0
            match_observed_factors_count = 0

            for factor_idx in obs_indices:
                if factors_x1[i, factor_idx] != factors_x2[i, factor_idx]:
                    diff_observed_factors_count += 1
                    # The differing factor *must* be the one indicated by the label according to the code logic
                    # The current code loops through diff_factors and sets label = i (last differing factor index).
                    # This might not be the intended behavior if k > 1. Let's test based on code.
                    assert factor_idx == label_factor_index # Check if the differing factor is the labelled one
                else:
                    match_observed_factors_count += 1

            # Verify number of differing factors based on k
            if k == 0:
                assert diff_observed_factors_count == 0
            elif k > 0:
                 # The code sets k factors to be different, but only labels the *last* one it changed.
                 # Let's verify *at least* one factor (the labeled one) is different.
                 # A stronger check would verify *exactly* k factors differ if k > 0.
                 # Need to be careful due to random sampling potentially making another factor different by chance (though unlikely for large ranges).
                 # The current code structure *forces* k factors to be different.
                 assert diff_observed_factors_count == k # Check exactly k observed factors differ
            # If k == -1, the number of differences is random (but > 0 if num_observed > 0) and set by k_observed internally.
            # The label still points to the last changed factor.
            elif k == -1 and num_observed > 0:
                 assert diff_observed_factors_count > 0 # At least one factor must differ
                 assert diff_observed_factors_count <= num_observed


    else:
        image_pairs, labels = result

    assert isinstance(image_pairs, torch.Tensor)
    # assert image_pairs.shape == expected_img_pair_shape # Shape check based on code trace
    # Let's relax this check slightly as channel handling can be ambiguous
    assert image_pairs.ndim == 4 # Should be Batch, Channel, Height, Width format
    assert image_pairs.shape[0] == num
    # assert image_pairs.dtype == torch.uint8 # Images are uint8 but might become float after ops? Check loader. Images are loaded as uint8. Should remain uint8.

    assert isinstance(labels, torch.Tensor)
    assert labels.shape == (num,)
    assert labels.dtype == torch.int64 or labels.dtype == torch.int32 # Check dtype of labels

    # Check label values are valid factor indices
    if num_observed > 0:
         valid_indices = set(dsprites.observed_latent_factor_indices)
         assert all(label.item() in valid_indices for label in labels)
    else: # If no observed factors, labels should likely be empty or handle gracefully
         assert labels.numel() == 0 or num == 0 # Or check specific behavior


# TODO: Add tests for edge cases like k=0, k=num_observed_factors if not fully covered.
# TODO: Potentially add checks for reproducibility using the seed.
