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
"""Callback and CallbackFactory unit tests."""

from unittest.mock import MagicMock

import pytest

from p2pfl.learning.frameworks import Framework
from p2pfl.learning.frameworks.callback import P2PFLCallback
from p2pfl.learning.frameworks.callback_factory import CallbackFactory

###############################################
# CallbackFactory Tests
###############################################


class MockCallback(P2PFLCallback):
    """Mock callback for testing."""

    @staticmethod
    def get_name() -> str:
        """Return the callback name."""
        return "mock_callback"


class AnotherMockCallback(P2PFLCallback):
    """Another mock callback for testing."""

    @staticmethod
    def get_name() -> str:
        """Return the callback name."""
        return "another_mock"


def test_callback_factory_register_and_create():
    """Test registering and creating callbacks."""
    # Register a mock callback
    CallbackFactory.register_callback(learner="test_framework", callback=MockCallback)

    # Create a mock aggregator that requires the callback
    mock_aggregator = MagicMock()
    mock_aggregator.get_required_callbacks.return_value = ["mock_callback"]

    callbacks = CallbackFactory.create_callbacks(framework="test_framework", aggregator=mock_aggregator)

    assert len(callbacks) == 1
    assert isinstance(callbacks[0], MockCallback)


def test_callback_factory_no_required_callbacks():
    """Test that empty list is returned when no callbacks are required."""
    mock_aggregator = MagicMock()
    mock_aggregator.get_required_callbacks.return_value = []

    callbacks = CallbackFactory.create_callbacks(framework=Framework.PYTORCH.value, aggregator=mock_aggregator)

    assert callbacks == []


def test_callback_factory_duplicate_registration_raises():
    """Test that registering the same callback twice raises an error."""
    # First registration should work
    CallbackFactory.register_callback(learner="duplicate_test", callback=AnotherMockCallback)

    # Second registration should raise
    with pytest.raises(ValueError, match="already registered"):
        CallbackFactory.register_callback(learner="duplicate_test", callback=AnotherMockCallback)


def test_callback_factory_unregistered_framework_raises():
    """Test that creating callbacks for an unregistered framework raises an error."""
    mock_aggregator = MagicMock()
    mock_aggregator.get_required_callbacks.return_value = ["some_callback"]

    with pytest.raises(ValueError, match="No callbacks registered"):
        CallbackFactory.create_callbacks(framework="nonexistent_framework", aggregator=mock_aggregator)


def test_callback_factory_missing_required_callback_raises():
    """Test that missing required callback raises an error."""
    # Register a framework with one callback
    CallbackFactory.register_callback(learner="partial_framework", callback=MockCallback)

    # But require a different callback
    mock_aggregator = MagicMock()
    mock_aggregator.get_required_callbacks.return_value = ["nonexistent_callback"]

    with pytest.raises(ValueError, match="not registered"):
        CallbackFactory.create_callbacks(framework="partial_framework", aggregator=mock_aggregator)


###############################################
# FedProx Callback Tests (PyTorch)
###############################################


def test_fedprox_callback_get_name():
    """Test FedProxCallback.get_name returns correct name."""
    from p2pfl.learning.frameworks.pytorch.callbacks.fedprox_callback import FedProxCallback

    callback = FedProxCallback()
    assert callback.get_name() == "fedprox"


def test_fedprox_callback_first_round_skips_proximal():
    """Test that FedProxCallback skips proximal term on first round."""
    from unittest.mock import MagicMock

    from p2pfl.learning.frameworks.pytorch.callbacks.fedprox_callback import FedProxCallback

    callback = FedProxCallback()
    mock_trainer = MagicMock()
    mock_module = MagicMock()

    # First round - should not raise and not set proximal_mu
    callback.on_train_start(mock_trainer, mock_module)

    assert callback.proximal_mu is None
    assert callback.initial_params is None
    assert callback._is_first_round is False  # Should be set to False after first call


def test_fedprox_callback_second_round_requires_info():
    """Test that FedProxCallback requires proximal_mu after first round."""
    from unittest.mock import MagicMock

    from p2pfl.learning.frameworks.pytorch.callbacks.fedprox_callback import FedProxCallback

    callback = FedProxCallback()
    mock_trainer = MagicMock()
    mock_module = MagicMock()

    # First call - sets _is_first_round to False
    callback.on_train_start(mock_trainer, mock_module)

    # Second call without additional_info should raise
    with pytest.raises(ValueError, match="proximal_mu required"):
        callback.on_train_start(mock_trainer, mock_module)


def test_fedprox_callback_second_round_with_info():
    """Test that FedProxCallback works correctly with proximal_mu set."""
    from unittest.mock import MagicMock

    import torch

    from p2pfl.learning.frameworks.pytorch.callbacks.fedprox_callback import FedProxCallback

    callback = FedProxCallback()
    mock_trainer = MagicMock()

    # Create a mock module with parameters
    mock_module = MagicMock()
    param1 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    param2 = torch.tensor([4.0, 5.0], requires_grad=True)
    mock_module.parameters.return_value = [param1, param2]

    # First call
    callback.on_train_start(mock_trainer, mock_module)

    # Set additional info for second round
    callback.additional_info = {"proximal_mu": 0.1}

    # Second call should work and snapshot parameters
    callback.on_train_start(mock_trainer, mock_module)

    assert callback.proximal_mu == 0.1
    assert callback.initial_params is not None
    assert len(callback.initial_params) == 2


###############################################
# Scaffold Callback Tests (PyTorch)
###############################################


def test_scaffold_callback_get_name():
    """Test SCAFFOLDCallback.get_name returns correct name."""
    from p2pfl.learning.frameworks.pytorch.callbacks.scaffold_callback import SCAFFOLDCallback

    callback = SCAFFOLDCallback()
    assert callback.get_name() == "scaffold"


def test_scaffold_callback_initial_state():
    """Test SCAFFOLDCallback initial state."""
    from p2pfl.learning.frameworks.pytorch.callbacks.scaffold_callback import SCAFFOLDCallback

    callback = SCAFFOLDCallback()

    assert callback.c == []
    assert callback.c_i == []
    assert callback.initial_model_params == []
