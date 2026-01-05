#
# This file is part of the p2pfl distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2024 Pedro Guijas Bravo.
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
Base aggregator tests - compatibility, validation, and custom aggregators.

Uses mock models to test aggregator framework without ML framework dependencies.
This allows testing the validation/compatibility system in isolation.
"""

import numpy as np
import pytest
from mocks import TreeBasedModelMock, WeightBasedModelMock

from p2pfl.learning.aggregators.aggregator import IncompatibleModelError, NoModelsToAggregateError, WeightAggregator
from p2pfl.learning.aggregators.fedavg import FedAvg
from p2pfl.learning.aggregators.sequential import SequentialLearning

###############################################
# Inheritance Hierarchy Tests
###############################################


def test_inheritance_hierarchy():
    """Test that aggregators inherit from correct base classes."""
    from p2pfl.learning.aggregators.aggregator import Aggregator

    assert issubclass(FedAvg, WeightAggregator)
    assert issubclass(SequentialLearning, Aggregator)


###############################################
# Model Compatibility Tests
###############################################


def test_weight_aggregator_accepts_weight_models():
    """Test that WeightAggregator accepts WeightBasedModel."""
    models = [
        WeightBasedModelMock(params=[np.array([1, 2, 3])], num_samples=1, contributors=["1"]),
        WeightBasedModelMock(params=[np.array([4, 5, 6])], num_samples=1, contributors=["2"]),
    ]
    aggregator = FedAvg()
    aggregator.set_addr("test")

    result = aggregator.aggregate(models)
    assert result is not None


def test_weight_aggregator_rejects_tree_models():
    """Test that WeightAggregator rejects TreeBasedModel."""
    models = [TreeBasedModelMock(params={"tree": "data"}, num_samples=1, contributors=["1"])]
    aggregator = FedAvg()
    aggregator.set_addr("test")

    with pytest.raises(IncompatibleModelError) as exc_info:
        aggregator.aggregate(models)

    assert "FedAvg" in str(exc_info.value)
    assert "TreeBasedModelMock" in str(exc_info.value)


def test_weight_aggregator_rejects_mixed_models():
    """Test that WeightAggregator rejects mixed model types."""
    models = [
        WeightBasedModelMock(params=[np.array([1, 2, 3])], num_samples=1, contributors=["1"]),
        TreeBasedModelMock(params={"tree": "data"}, num_samples=1, contributors=["2"]),
    ]
    aggregator = FedAvg()
    aggregator.set_addr("test")

    with pytest.raises(IncompatibleModelError):
        aggregator.aggregate(models)


###############################################
# SequentialLearning Tests
###############################################


def test_sequential_learning_accepts_weight_models():
    """Test that SequentialLearning accepts WeightBasedModel."""
    models = [WeightBasedModelMock(params=[np.array([1, 2, 3])], num_samples=10, contributors=["1"])]
    aggregator = SequentialLearning()
    aggregator.set_addr("test")

    result = aggregator.aggregate(models)
    assert result is not None
    assert result.get_num_samples() == 10
    assert result.get_contributors() == ["1"]


def test_sequential_learning_accepts_tree_models():
    """Test that SequentialLearning accepts TreeBasedModel."""
    models = [TreeBasedModelMock(params={"tree": "data"}, num_samples=20, contributors=["2"])]
    aggregator = SequentialLearning()
    aggregator.set_addr("test")

    result = aggregator.aggregate(models)
    assert result is not None
    assert result.get_num_samples() == 20
    assert result.get_contributors() == ["2"]


def test_sequential_learning_rejects_multiple_models():
    """Test that SequentialLearning rejects multiple models."""
    models = [
        WeightBasedModelMock(params=[np.array([1, 2, 3])], num_samples=1, contributors=["1"]),
        WeightBasedModelMock(params=[np.array([4, 5, 6])], num_samples=1, contributors=["2"]),
    ]
    aggregator = SequentialLearning()
    aggregator.set_addr("test")

    with pytest.raises(ValueError):
        aggregator.aggregate(models)


def test_aggregator_rejects_empty_list():
    """Test that aggregators reject empty model list."""
    aggregator = FedAvg()
    aggregator.set_addr("test")

    with pytest.raises(NoModelsToAggregateError):
        aggregator.aggregate([])


###############################################
# Aggregator Lifecycle Tests
###############################################


def test_aggregator_lifecycle():
    """Test aggregator lifecycle: set nodes, add models, wait, clear."""
    import time

    aggregator = FedAvg()
    aggregator.set_addr("test")
    aggregator.set_nodes_to_aggregate(["node1", "node2", "node3"])

    # Can't set nodes again while aggregating
    with pytest.raises(RuntimeError):
        aggregator.set_nodes_to_aggregate(["node4"])

    # Add first model
    model1 = WeightBasedModelMock([np.array([1.0])], num_samples=1, contributors=["node1"])
    aggregator.add_model(model1)

    assert not aggregator._finish_aggregation_event.is_set()
    assert aggregator.get_aggregated_models() == ["node1"]

    # Add remaining models
    model23 = WeightBasedModelMock([np.array([2.0])], num_samples=1, contributors=["node2", "node3"])
    aggregator.add_model(model23)

    assert set(aggregator.get_aggregated_models()) == {"node1", "node2", "node3"}

    # Wait should return immediately
    t = time.time()
    aggregator.wait_and_get_aggregation(timeout=1)
    assert time.time() - t < 1

    # Clear resets state
    aggregator.clear()
    assert aggregator.get_aggregated_models() == []


def test_partial_aggregation():
    """Test partial aggregation returns subset of models."""
    aggregator = FedAvg()
    aggregator.set_addr("test")
    aggregator.set_nodes_to_aggregate(["node1", "node2", "node3"])

    model1 = WeightBasedModelMock([np.array([1.0, 2.0])], num_samples=1, contributors=["node1"])
    aggregator.add_model(model1)

    model23 = WeightBasedModelMock([np.array([3.0, 4.0])], num_samples=1, contributors=["node2", "node3"])
    aggregator.add_model(model23)

    # Exclude node2/node3, should get only node1's model
    partial = aggregator.get_model(["node2", "node3"])
    assert np.allclose(partial.get_parameters()[0], model1.get_parameters()[0])
