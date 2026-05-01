#
# This file is part of the p2pfl distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2026 Pedro Guijas Bravo.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

"""Tests for Differential Privacy."""

import random

import numpy as np
import pytest

from p2pfl.learning.compression.dp_strategy import DifferentialPrivacyCompressor
from p2pfl.settings import Settings

try:
    from p2pfl.examples.mnist.model.mlp_pytorch import model_build_fn as model_build_fn_torch
except ImportError:
    model_build_fn_torch = pytest.param(None, marks=pytest.mark.skip(reason="PyTorch not installed"))  # type: ignore[assignment]


###
# Differential Privacy Compressor tests
###


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    Settings.general.SEED = seed
    yield


@pytest.fixture
def dp_compressor():
    """Create a DifferentialPrivacyCompressor instance."""
    return DifferentialPrivacyCompressor()


def test_dp_basic_sanity(dp_compressor):
    """
    Sanity Check for DifferentialPrivacyCompressor.

    - Code does not crash.
    - Output shapes match input shapes.
    - 'dp_applied' flag in metadata is True.
    """
    # Prepare simple parameters: two arrays of different shapes
    original_params = [np.ones((3, 4), dtype=np.float32), np.zeros((2, 2), dtype=np.float32)]

    # Apply DP strategy with example privacy settings
    dp_params, info = dp_compressor.apply_strategy(params=original_params, clip_norm=1.0, epsilon=4.0, delta=1e-5, noise_type="gaussian")

    # 1) Ensure we got a list of numpy arrays back
    assert isinstance(dp_params, list), "Expected a list of arrays as output"
    assert all(isinstance(p, np.ndarray) for p in dp_params), "Each element must be a numpy.ndarray"

    # 2) Shapes must be preserved
    expected_shapes = [p.shape for p in original_params]
    actual_shapes = [p.shape for p in dp_params]
    assert actual_shapes == expected_shapes, f"Expected shapes {expected_shapes}, got {actual_shapes}"

    # 3) DP must be applied according to metadata
    assert info.get("dp_applied", False) is True, "Expected 'dp_applied' to be True"

    # 4) Basic metadata keys must exist
    for key in ("clip_norm", "epsilon", "noise_type", "noise_scale", "original_norm", "was_clipped"):
        assert key in info, f"Missing metadata key: '{key}'"


def test_dp_clipping_when_needed(dp_compressor):
    """
    Test that updates with norm > clip_norm are clipped.

    - original_norm should be greater than clip_norm
    - was_clipped flag must be True
    """
    # Prepare a single parameter array whose norm is definitely > clip_norm
    # e.g. a 2x2 array of all 10s has norm = 10 * sqrt(4) = 20
    original_params = [np.full((2, 2), 10.0, dtype=np.float32)]
    clip_norm = 1.0

    dp_params, info = dp_compressor.apply_strategy(
        params=original_params, clip_norm=clip_norm, epsilon=4.0, delta=1e-5, noise_type="gaussian"
    )

    # 1) original_norm should exceed clip_norm
    assert info["original_norm"] > clip_norm, f"Expected original_norm > {clip_norm}, got {info['original_norm']}"
    # 2) was_clipped must be True
    assert info["was_clipped"] is True, "Expected was_clipped to be True when original_norm > clip_norm"


def test_dp_no_clipping_when_not_needed(dp_compressor):
    """
    Test that updates with norm <= clip_norm are not clipped.

    - original_norm should be less than or equal to clip_norm
    - was_clipped flag must be False
    """
    # Prepare a single parameter array whose norm is definitely < clip_norm
    # e.g. a 2x2 array of all 1s has norm = 1 * sqrt(4) = 2
    original_params = [np.ones((2, 2), dtype=np.float32)]
    clip_norm = 10.0

    dp_params, info = dp_compressor.apply_strategy(
        params=original_params, clip_norm=clip_norm, epsilon=4.0, delta=1e-5, noise_type="gaussian"
    )

    # 1) original_norm should not exceed clip_norm
    assert info["original_norm"] <= clip_norm, f"Expected original_norm <= {clip_norm}, got {info['original_norm']}"
    # 2) was_clipped must be False
    assert info["was_clipped"] is False, "Expected was_clipped to be False when original_norm <= clip_norm"


def test_dp_noise_addition(dp_compressor):
    """
    Test that noise is actually added.

    - Without clipping (clip_norm large), output must differ from input due to noise.
    """
    # Prepare a constant array (no clipping will occur if clip_norm >> norm)
    original_params = [np.full((4, 4), 5.0, dtype=np.float32)]
    clip_norm = 100.0

    dp_params, info = dp_compressor.apply_strategy(
        params=original_params, clip_norm=clip_norm, epsilon=4.0, delta=1e-5, noise_type="gaussian"
    )

    # Ensure DP was applied
    assert info.get("dp_applied", False) is True, "Expected noise to be applied (dp_applied=True)"

    # The output must differ from the input because of added noise
    assert not np.array_equal(dp_params[0], original_params[0]), "Expected dp_params to differ from original_params due to noise"


def test_dp_empty_params(dp_compressor):
    """
    Test that an empty parameter list raises a ValueError.

    - Input [] must raise a ValueError.
    """
    with pytest.raises(ValueError, match="must not be empty"):
        dp_compressor.apply_strategy(params=[], clip_norm=1.0, epsilon=4.0, delta=1e-5, noise_type="gaussian")
