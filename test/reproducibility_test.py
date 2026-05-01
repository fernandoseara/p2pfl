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
"""

Reproducibility tests.

These tests are not super important, but they are a good way to check that the code is reproducible.

Not recomended to run always, as they are slow.
"""

import numpy as np
import pytest  # noqa: E402, I001
from datasets import DatasetDict, load_dataset  # noqa: E402, I001

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset  # noqa: E402
from p2pfl.learning.dataset.partition_strategies import DirichletPartitionStrategy, RandomIIDPartitionStrategy
from p2pfl.settings import Settings

try:
    from p2pfl.examples.mnist.model.mlp_tensorflow import model_build_fn as model_build_fn_tensorflow
except ImportError:
    model_build_fn_tensorflow = pytest.param(None, marks=pytest.mark.skip(reason="TensorFlow not installed"))  # type: ignore[assignment]

try:
    from p2pfl.examples.mnist.model.mlp_pytorch import model_build_fn as model_build_fn_pytorch
except ImportError:
    model_build_fn_pytorch = pytest.param(None, marks=pytest.mark.skip(reason="PyTorch not installed"))  # type: ignore[assignment]

###
# Dataset Partitioning
###


@pytest.mark.parametrize(
    "strategy, strategy_kwargs",
    [
        (RandomIIDPartitionStrategy, {}),
        (DirichletPartitionStrategy, {"alpha": 0.5}),
    ],
)
def test_set_dataset_partition_reproducibility(strategy, strategy_kwargs):
    """Test that seed ensures reproducibility for partitioning strategies."""
    # Dataset
    mnist_dataset = P2PFLDataset(
        DatasetDict(
            {
                "train": load_dataset("p2pfl/MNIST", split="train[:100]"),
                "test": load_dataset("p2pfl/MNIST", split="test[:10]"),
            }
        )
    )
    # Test 1: Same global seed, same strategy seed -> same partitions
    Settings.general.SEED = 666
    partitions1 = mnist_dataset.generate_partitions(num_partitions=3, strategy=strategy, **strategy_kwargs)
    Settings.general.SEED = 666
    partitions2 = mnist_dataset.generate_partitions(num_partitions=3, strategy=strategy, **strategy_kwargs)

    # Verify partitions are the same
    for i in range(3):
        # Compare train data indices
        assert (
            partitions1[i]._data[partitions1[i]._train_split_name]._indices
            == partitions2[i]._data[partitions2[i]._train_split_name]._indices
        )

        # Compare test data indices
        assert (
            partitions1[i]._data[partitions1[i]._test_split_name]._indices == partitions2[i]._data[partitions2[i]._test_split_name]._indices
        )

    # Test 2: Different strategy seed -> different partitions
    Settings.general.SEED = 777
    partitions3 = mnist_dataset.generate_partitions(num_partitions=3, strategy=strategy, **strategy_kwargs)

    # Verify at least one partition is different
    different_strategy_seed = False
    for i in range(3):
        if (
            partitions1[i]._data[partitions1[i]._train_split_name]._indices
            != partitions3[i]._data[partitions3[i]._train_split_name]._indices
        ):
            different_strategy_seed = True
            break

    assert different_strategy_seed, "Partitions should be different with different strategy seeds"


###
# Model
###


@pytest.mark.parametrize("model_build_fn", [model_build_fn_pytorch, model_build_fn_tensorflow])  # , model_build_fn_flax])
def test_model_initialization_reproducibility(model_build_fn):
    """Test that seed ensures reproducible model initialization."""
    try:
        # First initialization with seed
        Settings.general.SEED = 666
        params1 = model_build_fn().get_parameters()

        # Second initialization with same seed
        Settings.general.SEED = 666
        params2 = model_build_fn().get_parameters()

        # Assert parameters are identical
        for p1, p2 in zip(params1, params2, strict=True):
            assert np.array_equal(p1, p2), "Model parameters differ despite using the same seed"

        # Different seed should produce different parameters
        Settings.general.SEED = 777
        params3 = model_build_fn().get_parameters()

        # At least one parameter should be different
        any_different = False
        for p1, p3 in zip(params1, params3, strict=True):
            if not np.array_equal(p1, p3):
                any_different = True
                break
        assert any_different, "Different seeds produced identical model parameters"

    except ImportError:
        pytest.skip("PyTorch not available")
