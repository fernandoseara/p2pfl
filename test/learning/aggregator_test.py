#
# This file is part of the federated_learning_p2p (p2pfl) distribution
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
"""Learning tests."""

import contextlib
import time
from typing import Any

import numpy as np
import pytest

from p2pfl.learning.aggregators.aggregator import Aggregator, IncompatibleModelError, compatible_with
from p2pfl.learning.aggregators.fedavg import FedAvg
from p2pfl.learning.aggregators.fedxgbcyclic import FedXgbCyclic
from p2pfl.learning.frameworks import ModelType
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel

# Import PyTorch models if available
with contextlib.suppress(ImportError):
    from p2pfl.examples.mnist.model.mlp_pytorch import MLP
    from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel


class P2PFLModelMock(P2PFLModel):
    """Mock model for testing purposes."""

    def __init__(
        self,
        model: Any,
        params: list[np.ndarray] | bytes | None = None,
        num_samples: int | None = None,
        contributors: list[str] | None = None,
        aditional_info: dict[str, str] | None = None,
        model_type: str = "neural_network",
    ) -> None:
        """Initialize the model."""
        self.params = params
        self.num_samples = num_samples  # type: ignore
        self.contributors = contributors  # type: ignore
        self._model_type = model_type

    def get_parameters(self):
        """Get the model parameters."""
        return self.params

    def get_num_samples(self):
        """Get the number of samples."""
        return self.num_samples

    def build_copy(self, **kwargs):
        """Build a copy of the model."""
        model_type = kwargs.pop("model_type", self._model_type)
        return P2PFLModelMock(None, model_type=model_type, **kwargs)

    def get_contributors(self) -> list[str]:
        """Get the contributors."""
        return self.contributors

    def get_model_type(self) -> str:
        """Get the model type for compatibility validation."""
        return self._model_type


def test_avg_simple():
    """Test simple aggregation (simple arrays)."""
    models = [
        P2PFLModelMock(None, params=[np.array([1, 2, 3])], num_samples=1, contributors=["1"]),
        P2PFLModelMock(None, params=[np.array([4, 5, 6])], num_samples=1, contributors=["2"]),
        P2PFLModelMock(None, params=[np.array([7, 8, 9])], num_samples=1, contributors=["3"]),
    ]
    # New aggregator test
    aggregator = FedAvg()
    aggregator.set_addr("prueba")
    res = aggregator.aggregate(models)

    assert np.array_equal(res.get_parameters()[0], np.array([4, 5, 6]))
    assert set(res.get_contributors()) == {"1", "2", "3"}


def test_avg_complex():
    """Test complex aggregation (models)."""
    # Initial Model
    model = LightningModel(MLP(), num_samples=1, contributors=["1"])

    params = model.get_parameters()

    # Model 1
    params1 = []
    for layer in params:
        params1.append(layer + 1.0)

    # Model 2
    params2 = []
    for layer in params:
        params2.append(layer - 1.0)

    # New aggregator test
    aggregator = FedAvg()
    aggregator.set_addr("prueba")
    res = aggregator.aggregate(
        [
            model,
            LightningModel(MLP(), params=params1, num_samples=2, contributors=["2"]),
            LightningModel(MLP(), params=params2, num_samples=2, contributors=["3"]),
        ]
    )

    # Assertion: Check if the aggregated parameters are equal to the initial model's parameters
    for i, layer in enumerate(res.get_parameters()):
        assert np.allclose(layer, model.get_parameters()[i], atol=1e-7), f"Layer {i} does not match"
    assert set(res.get_contributors()) == {"1", "2", "3"}


def test_aggregator_lifecycle():
    """Test the aggregator lock."""
    aggregator = FedAvg()
    aggregator.set_addr("prueba")
    aggregator.set_nodes_to_aggregate(["node1", "node2", "node3"])

    # Try to set nodes again (should raise an exception)
    with pytest.raises(Exception) as _:
        aggregator.set_nodes_to_aggregate(["node4"])

    # Add a model
    model1 = LightningModel(MLP(), num_samples=1, contributors=["node1"])
    aggregator.add_model(model1)

    # Ensure that the previous lock, now an event is cleared (equivalent to locked)
    assert not aggregator._finish_aggregation_event.is_set()

    # Check if the model was added
    assert aggregator.get_aggregated_models() == ["node1"]

    # Add the rest of the models
    model23 = LightningModel(MLP(), num_samples=1, contributors=["node2", "node3"])
    aggregator.add_model(model23)

    # Get partial aggregation
    partial_model = aggregator.get_model(["node2", "node3"])
    assert all((partial_model.get_parameters()[i] == model1.get_parameters()[i]).all() for i in range(len(partial_model.get_parameters())))

    # Check if the model was added
    assert set(aggregator.get_aggregated_models()) == {"node1", "node2", "node3"}

    # Ensure that the lock is released
    t = time.time()
    aggregator.wait_and_get_aggregation(timeout=1)
    assert time.time() - t < 1

    # Check clear
    aggregator.clear()
    assert aggregator.get_aggregated_models() == []


###############################################
# Model Compatibility Tests
###############################################


def test_compatible_with_decorator_sets_attribute():
    """Test that the @compatible_with decorator sets the COMPATIBLE_MODEL_TYPES attribute."""
    assert hasattr(FedAvg, "COMPATIBLE_MODEL_TYPES")
    assert ModelType.NEURAL_NETWORK in FedAvg.COMPATIBLE_MODEL_TYPES

    assert hasattr(FedXgbCyclic, "COMPATIBLE_MODEL_TYPES")
    assert ModelType.BOOSTING_TREE in FedXgbCyclic.COMPATIBLE_MODEL_TYPES


def test_fedavg_accepts_neural_network_models():
    """Test that FedAvg accepts neural network models."""
    models = [
        P2PFLModelMock(None, params=[np.array([1, 2, 3])], num_samples=1, contributors=["1"], model_type="neural_network"),
        P2PFLModelMock(None, params=[np.array([4, 5, 6])], num_samples=1, contributors=["2"], model_type="neural_network"),
    ]
    aggregator = FedAvg()
    aggregator.set_addr("test")

    # Should not raise any exception
    result = aggregator.aggregate(models)
    assert result is not None


def test_fedavg_rejects_boosting_tree_models():
    """Test that FedAvg rejects boosting tree models."""
    models = [
        P2PFLModelMock(None, params=[np.array([1, 2, 3])], num_samples=1, contributors=["1"], model_type="boosting_tree"),
    ]
    aggregator = FedAvg()
    aggregator.set_addr("test")

    with pytest.raises(IncompatibleModelError) as exc_info:
        aggregator.aggregate(models)

    assert "FedAvg" in str(exc_info.value)
    assert "boosting_tree" in str(exc_info.value)
    assert "neural_network" in str(exc_info.value)


def test_fedavg_rejects_mixed_models():
    """Test that FedAvg rejects when mixing neural network and boosting tree models."""
    models = [
        P2PFLModelMock(None, params=[np.array([1, 2, 3])], num_samples=1, contributors=["1"], model_type="neural_network"),
        P2PFLModelMock(None, params=[np.array([4, 5, 6])], num_samples=1, contributors=["2"], model_type="boosting_tree"),
    ]
    aggregator = FedAvg()
    aggregator.set_addr("test")

    with pytest.raises(IncompatibleModelError) as exc_info:
        aggregator.aggregate(models)

    assert "boosting_tree" in str(exc_info.value)


def test_fedxgbcyclic_accepts_boosting_tree_models():
    """Test that FedXgbCyclic accepts boosting tree models."""
    models = [
        P2PFLModelMock(None, params=[np.array([1, 2, 3])], num_samples=1, contributors=["1"], model_type="boosting_tree"),
    ]
    aggregator = FedXgbCyclic()
    aggregator.set_addr("test")

    # Should not raise any exception
    result = aggregator.aggregate(models)
    assert result is not None


def test_fedxgbcyclic_rejects_neural_network_models():
    """Test that FedXgbCyclic rejects neural network models."""
    models = [
        P2PFLModelMock(None, params=[np.array([1, 2, 3])], num_samples=1, contributors=["1"], model_type="neural_network"),
    ]
    aggregator = FedXgbCyclic()
    aggregator.set_addr("test")

    with pytest.raises(IncompatibleModelError) as exc_info:
        aggregator.aggregate(models)

    assert "FedXgbCyclic" in str(exc_info.value)
    assert "neural_network" in str(exc_info.value)
    assert "boosting_tree" in str(exc_info.value)


def test_custom_aggregator_without_decorator_accepts_any_model():
    """Test that an aggregator without @compatible_with accepts any model type."""

    class CustomAggregator(Aggregator):
        """Custom aggregator without compatibility restrictions."""

        def aggregate(self, models: list[P2PFLModel]) -> P2PFLModel:
            return models[0].build_copy(
                params=models[0].get_parameters(),
                num_samples=models[0].get_num_samples(),
                contributors=models[0].get_contributors(),
            )

    models_nn = [
        P2PFLModelMock(None, params=[np.array([1, 2, 3])], num_samples=1, contributors=["1"], model_type="neural_network"),
    ]
    models_bt = [
        P2PFLModelMock(None, params=[np.array([1, 2, 3])], num_samples=1, contributors=["1"], model_type="boosting_tree"),
    ]

    aggregator = CustomAggregator()
    aggregator.set_addr("test")

    # Both should work without exception
    result_nn = aggregator.aggregate(models_nn)
    assert result_nn is not None

    result_bt = aggregator.aggregate(models_bt)
    assert result_bt is not None


def test_custom_aggregator_with_multiple_model_types():
    """Test that an aggregator can be compatible with multiple model types."""

    @compatible_with(ModelType.NEURAL_NETWORK, ModelType.BOOSTING_TREE)
    class MultiCompatibleAggregator(Aggregator):
        """Aggregator compatible with both neural networks and boosting trees."""

        def aggregate(self, models: list[P2PFLModel]) -> P2PFLModel:
            return models[0].build_copy(
                params=models[0].get_parameters(),
                num_samples=models[0].get_num_samples(),
                contributors=models[0].get_contributors(),
            )

    models_nn = [
        P2PFLModelMock(None, params=[np.array([1, 2, 3])], num_samples=1, contributors=["1"], model_type="neural_network"),
    ]
    models_bt = [
        P2PFLModelMock(None, params=[np.array([1, 2, 3])], num_samples=1, contributors=["1"], model_type="boosting_tree"),
    ]

    aggregator = MultiCompatibleAggregator()
    aggregator.set_addr("test")

    # Both should work
    result_nn = aggregator.aggregate(models_nn)
    assert result_nn is not None

    result_bt = aggregator.aggregate(models_bt)
    assert result_bt is not None


def test_incompatible_model_error_message():
    """Test that IncompatibleModelError provides a clear error message."""
    models = [
        P2PFLModelMock(None, params=[np.array([1, 2, 3])], num_samples=1, contributors=["1"], model_type="boosting_tree"),
    ]
    aggregator = FedAvg()
    aggregator.set_addr("test")

    with pytest.raises(IncompatibleModelError) as exc_info:
        aggregator.aggregate(models)

    error_message = str(exc_info.value)
    # Check that the error message contains useful information
    assert "FedAvg" in error_message
    assert "boosting_tree" in error_message
    assert "neural_network" in error_message
    assert "not compatible" in error_message.lower() or "incompatible" in error_message.lower()
