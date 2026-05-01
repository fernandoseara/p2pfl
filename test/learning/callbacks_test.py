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

import contextlib
from unittest.mock import MagicMock

import pytest

from p2pfl.learning.frameworks import Framework
from p2pfl.learning.frameworks.callback import P2PFLCallback
from p2pfl.learning.frameworks.callback_factory import CallbackFactory

###
# CallbackFactory Tests
###


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


###
# FedProx Callback Tests (PyTorch)
###


def test_fedprox_callback_get_name():
    """Test FedProxCallback.get_name returns correct name."""
    pytest.importorskip("lightning", reason="PyTorch Lightning not available")
    from p2pfl.learning.frameworks.pytorch.callbacks.fedprox_callback import FedProxCallback

    callback = FedProxCallback()
    assert callback.get_name() == "fedprox"


def test_fedprox_callback_first_round_skips_proximal():
    """Test that FedProxCallback skips proximal term on first round."""
    pytest.importorskip("lightning", reason="PyTorch Lightning not available")
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
    pytest.importorskip("lightning", reason="PyTorch Lightning not available")
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
    pytest.importorskip("torch", reason="PyTorch not available")
    pytest.importorskip("lightning", reason="PyTorch Lightning not available")
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


###
# Scaffold Callback Tests (PyTorch)
###


def test_scaffold_pt_on_train_start_initializes_control_variates_as_zeros():
    """Test that on_train_start initializes c_i and c as zero tensors matching model params."""
    torch = pytest.importorskip("torch", reason="PyTorch not available")
    pytest.importorskip("lightning", reason="PyTorch Lightning not available")
    from p2pfl.learning.frameworks.pytorch.callbacks.scaffold_callback import SCAFFOLDCallback

    callback = SCAFFOLDCallback()

    mock_module = MagicMock()
    mock_module.state_dict.return_value = {
        "layer1.weight": torch.tensor([1.0, 2.0, 3.0]),
        "layer1.bias": torch.tensor([0.5]),
    }
    mock_trainer = MagicMock()

    callback.on_train_start(mock_trainer, mock_module)

    assert len(callback.c_i) == 2
    assert len(callback.c) == 2
    assert torch.equal(callback.c_i[0], torch.zeros(3))
    assert torch.equal(callback.c_i[1], torch.zeros(1))
    assert torch.equal(callback.c[0], torch.zeros(3))
    assert torch.equal(callback.c[1], torch.zeros(1))


def test_scaffold_pt_on_train_start_loads_global_c_from_additional_info():
    """Test that on_train_start loads global control variate from additional_info."""
    torch = pytest.importorskip("torch", reason="PyTorch not available")
    np = pytest.importorskip("numpy", reason="NumPy not available")
    pytest.importorskip("lightning", reason="PyTorch Lightning not available")
    from p2pfl.learning.frameworks.pytorch.callbacks.scaffold_callback import SCAFFOLDCallback

    callback = SCAFFOLDCallback()

    mock_module = MagicMock()
    mock_module.state_dict.return_value = {
        "w": torch.tensor([1.0, 2.0]),
    }
    mock_module.device = torch.device("cpu")
    mock_trainer = MagicMock()

    # Provide global_c via additional_info (float32 to match torch default)
    global_c = [np.array([0.1, 0.2], dtype=np.float32)]
    callback.additional_info = {"global_c": global_c}

    callback.on_train_start(mock_trainer, mock_module)

    assert len(callback.c) == 1
    assert torch.allclose(callback.c[0], torch.tensor([0.1, 0.2]))


def test_scaffold_pt_on_train_start_preserves_existing_c_i():
    """Test that on_train_start does not re-initialize c_i if already set."""
    torch = pytest.importorskip("torch", reason="PyTorch not available")
    pytest.importorskip("lightning", reason="PyTorch Lightning not available")
    from p2pfl.learning.frameworks.pytorch.callbacks.scaffold_callback import SCAFFOLDCallback

    callback = SCAFFOLDCallback()
    # Pre-set c_i with non-zero values
    callback.c_i = [torch.tensor([5.0, 6.0])]

    mock_module = MagicMock()
    mock_module.state_dict.return_value = {"w": torch.tensor([1.0, 2.0])}
    mock_trainer = MagicMock()

    callback.on_train_start(mock_trainer, mock_module)

    # c_i should NOT have been overwritten
    assert torch.equal(callback.c_i[0], torch.tensor([5.0, 6.0]))


def test_scaffold_pt_gradient_correction():
    """Test that on_before_zero_grad applies SCAFFOLD gradient correction: grad += lr * (c_i - c)."""
    torch = pytest.importorskip("torch", reason="PyTorch not available")
    pytest.importorskip("lightning", reason="PyTorch Lightning not available")
    from p2pfl.learning.frameworks.pytorch.callbacks.scaffold_callback import SCAFFOLDCallback

    callback = SCAFFOLDCallback()
    callback.saved_lr = 0.1
    callback.c_i = [torch.tensor([1.0, 2.0])]
    callback.c = [torch.tensor([0.5, 0.5])]

    # Create a module mock with a parameter that has a gradient
    param = torch.tensor([10.0, 20.0])
    param.grad = torch.tensor([1.0, 1.0])

    mock_module = MagicMock()
    mock_module.state_dict.return_value = {"w": param}

    mock_trainer = MagicMock()
    mock_optimizer = MagicMock()

    callback.on_before_zero_grad(mock_trainer, mock_module, mock_optimizer)

    # grad should be: [1.0, 1.0] + 0.1 * ([1.0, 2.0] - [0.5, 0.5]) = [1.0, 1.0] + [0.05, 0.15] = [1.05, 1.15]
    expected = torch.tensor([1.05, 1.15])
    assert torch.allclose(param.grad, expected)
    assert callback.K == 1


def test_scaffold_pt_on_train_end_computes_delta_c_correctly():
    """Test on_train_end computes delta_c_i = new_c_i - old_c_i following the SCAFFOLD formula."""
    torch = pytest.importorskip("torch", reason="PyTorch not available")
    np = pytest.importorskip("numpy", reason="NumPy not available")
    pytest.importorskip("lightning", reason="PyTorch Lightning not available")
    from p2pfl.learning.frameworks.pytorch.callbacks.scaffold_callback import SCAFFOLDCallback

    callback = SCAFFOLDCallback()
    callback.saved_lr = 0.1
    callback.K = 5

    # c_i starts at [0.0, 0.0]
    callback.c_i = [torch.tensor([0.0, 0.0])]
    # initial model params (x_g): [1.0, 2.0]
    callback.initial_model_params = [torch.tensor([1.0, 2.0])]

    # Current model params (y_i) after training: [0.5, 1.5]
    mock_module = MagicMock()
    mock_module.state_dict.return_value = {"w": torch.tensor([0.5, 1.5])}

    mock_trainer = MagicMock()

    callback.on_train_end(mock_trainer, mock_module)

    # adjustment = (x_g - y_i) / (K * lr) = ([1.0, 2.0] - [0.5, 1.5]) / (5 * 0.1) = [0.5, 0.5] / 0.5 = [1.0, 1.0]
    # new_c_i = old_c_i + adjustment = [0.0, 0.0] + [1.0, 1.0] = [1.0, 1.0]
    # delta_c_i = new_c_i - old_c_i = [1.0, 1.0] - [0.0, 0.0] = [1.0, 1.0]
    assert "delta_c_i" in callback.additional_info
    assert "delta_y_i" in callback.additional_info

    delta_c_i = callback.additional_info["delta_c_i"]
    assert len(delta_c_i) == 1
    np.testing.assert_allclose(delta_c_i[0], [1.0, 1.0])

    # delta_y_i = y_i - x_g = [0.5, 1.5] - [1.0, 2.0] = [-0.5, -0.5]
    delta_y_i = callback.additional_info["delta_y_i"]
    np.testing.assert_allclose(delta_y_i[0], [-0.5, -0.5])


def test_scaffold_pt_on_train_end_updates_c_i():
    """Test on_train_end correctly updates c_i with the adjustment term."""
    torch = pytest.importorskip("torch", reason="PyTorch not available")
    pytest.importorskip("lightning", reason="PyTorch Lightning not available")
    from p2pfl.learning.frameworks.pytorch.callbacks.scaffold_callback import SCAFFOLDCallback

    callback = SCAFFOLDCallback()
    callback.saved_lr = 0.5
    callback.K = 2

    # c_i starts at [0.3, 0.7]
    callback.c_i = [torch.tensor([0.3, 0.7])]
    callback.initial_model_params = [torch.tensor([2.0, 4.0])]

    # y_i after training: [1.0, 3.0]
    mock_module = MagicMock()
    mock_module.state_dict.return_value = {"w": torch.tensor([1.0, 3.0])}
    mock_trainer = MagicMock()

    callback.on_train_end(mock_trainer, mock_module)

    # adjustment = (x_g - y_i) / (K * lr) = ([2.0, 4.0] - [1.0, 3.0]) / (2 * 0.5) = [1.0, 1.0] / 1.0 = [1.0, 1.0]
    # new_c_i = [0.3, 0.7] + [1.0, 1.0] = [1.3, 1.7]
    assert torch.allclose(callback.c_i[0], torch.tensor([1.3, 1.7]))


def test_scaffold_pt_on_train_end_raises_without_init():
    """Test on_train_end raises if attributes are not initialized."""
    pytest.importorskip("torch", reason="PyTorch not available")
    pytest.importorskip("lightning", reason="PyTorch Lightning not available")
    from p2pfl.learning.frameworks.pytorch.callbacks.scaffold_callback import SCAFFOLDCallback

    callback = SCAFFOLDCallback()
    mock_trainer = MagicMock()
    mock_module = MagicMock()

    with pytest.raises(AttributeError, match="Necessary attributes are not initialized"):
        callback.on_train_end(mock_trainer, mock_module)


def test_scaffold_pt_gradient_correction_raises_without_lr():
    """Test on_before_zero_grad raises if learning rate is not set."""
    pytest.importorskip("torch", reason="PyTorch not available")
    pytest.importorskip("lightning", reason="PyTorch Lightning not available")
    from p2pfl.learning.frameworks.pytorch.callbacks.scaffold_callback import SCAFFOLDCallback

    callback = SCAFFOLDCallback()
    # saved_lr is None by default
    mock_trainer = MagicMock()
    mock_module = MagicMock()
    mock_optimizer = MagicMock()

    with pytest.raises(AttributeError, match="Learning rate has not been set"):
        callback.on_before_zero_grad(mock_trainer, mock_module, mock_optimizer)


def test_scaffold_pt_multiple_params():
    """Test SCAFFOLD delta computation with multiple parameter groups."""
    torch = pytest.importorskip("torch", reason="PyTorch not available")
    np = pytest.importorskip("numpy", reason="NumPy not available")
    pytest.importorskip("lightning", reason="PyTorch Lightning not available")
    from p2pfl.learning.frameworks.pytorch.callbacks.scaffold_callback import SCAFFOLDCallback

    callback = SCAFFOLDCallback()
    callback.saved_lr = 0.01
    callback.K = 10

    callback.c_i = [torch.tensor([0.0]), torch.tensor([0.0, 0.0])]
    callback.initial_model_params = [torch.tensor([1.0]), torch.tensor([2.0, 3.0])]

    # y_i after training
    mock_module = MagicMock()
    mock_module.state_dict.return_value = {
        "layer1.weight": torch.tensor([0.9]),
        "layer1.bias": torch.tensor([1.8, 2.7]),
    }
    mock_trainer = MagicMock()

    callback.on_train_end(mock_trainer, mock_module)

    # param 0: adjustment = (1.0 - 0.9) / (10 * 0.01) = 0.1 / 0.1 = 1.0
    # param 1: adjustment = ([2.0, 3.0] - [1.8, 2.7]) / (10 * 0.01) = [0.2, 0.3] / 0.1 = [2.0, 3.0]
    delta_c_i = callback.additional_info["delta_c_i"]
    np.testing.assert_allclose(delta_c_i[0], [1.0], atol=1e-6)
    np.testing.assert_allclose(delta_c_i[1], [2.0, 3.0], atol=1e-6)

    delta_y_i = callback.additional_info["delta_y_i"]
    np.testing.assert_allclose(delta_y_i[0], [-0.1], atol=1e-6)
    np.testing.assert_allclose(delta_y_i[1], [-0.2, -0.3], atol=1e-6)


###
# Scaffold Callback Tests (TensorFlow)
###


def test_scaffold_tf_get_name():
    """Test TF SCAFFOLDCallback.get_name returns correct name."""
    pytest.importorskip("tensorflow", reason="TensorFlow not available")
    from p2pfl.learning.frameworks.tensorflow.callbacks.scaffold_callback import SCAFFOLDCallback

    callback = SCAFFOLDCallback()
    assert callback.get_name() == "scaffold"


def test_scaffold_tf_initial_state():
    """Test TF SCAFFOLDCallback initial state."""
    pytest.importorskip("tensorflow", reason="TensorFlow not available")
    from p2pfl.learning.frameworks.tensorflow.callbacks.scaffold_callback import SCAFFOLDCallback

    callback = SCAFFOLDCallback()
    assert callback.c == []
    assert callback.c_i == []
    assert callback.initial_model_params == []
    assert callback.saved_lr is None
    assert callback.K == 0


def test_scaffold_tf_on_train_begin_initializes_control_variates():
    """Test TF on_train_begin initializes c_i and c as zeros."""
    tf = pytest.importorskip("tensorflow", reason="TensorFlow not available")
    from p2pfl.learning.frameworks.tensorflow.callbacks.scaffold_callback import SCAFFOLDCallback

    callback = SCAFFOLDCallback()

    # Set up mock model with trainable variables and optimizer
    param1 = tf.Variable([1.0, 2.0, 3.0])
    param2 = tf.Variable([0.5])
    mock_model = MagicMock()
    mock_model.trainable_variables = [param1, param2]

    mock_optimizer = MagicMock()
    mock_optimizer.learning_rate = 0.01
    mock_model.optimizer = mock_optimizer

    callback.set_model(mock_model)

    # ScaffoldOptimizerWrapper cannot be instantiated with Keras 3 due to
    # __setattr__/__getattr__ recursion in the Optimizer base class.
    # We test the init logic up to that point.
    with contextlib.suppress(RecursionError):
        callback.on_train_begin()

    assert len(callback.c_i) == 2
    assert len(callback.c) == 2
    # Check zeros
    assert float(tf.reduce_sum(tf.abs(callback.c_i[0]))) == 0.0
    assert float(tf.reduce_sum(tf.abs(callback.c_i[1]))) == 0.0
    assert float(tf.reduce_sum(tf.abs(callback.c[0]))) == 0.0
    assert float(tf.reduce_sum(tf.abs(callback.c[1]))) == 0.0
    assert callback.saved_lr == pytest.approx(0.01)
    assert callback.K == 0


def test_scaffold_tf_on_train_begin_loads_global_c():
    """Test TF on_train_begin loads global_c from additional_info."""
    tf = pytest.importorskip("tensorflow", reason="TensorFlow not available")
    np = pytest.importorskip("numpy", reason="NumPy not available")
    from p2pfl.learning.frameworks.tensorflow.callbacks.scaffold_callback import SCAFFOLDCallback

    callback = SCAFFOLDCallback()

    param = tf.Variable([1.0, 2.0])
    mock_model = MagicMock()
    mock_model.trainable_variables = [param]
    mock_optimizer = MagicMock()
    mock_optimizer.learning_rate = 0.1
    mock_model.optimizer = mock_optimizer

    callback.set_model(mock_model)
    callback.additional_info = {"global_c": [np.array([0.3, 0.7])]}

    with contextlib.suppress(RecursionError):
        callback.on_train_begin()

    assert len(callback.c) == 1
    np.testing.assert_allclose(callback.c[0].numpy(), [0.3, 0.7])


def test_scaffold_tf_on_train_batch_end_increments_K():
    """Test TF on_train_batch_end increments K."""
    pytest.importorskip("tensorflow", reason="TensorFlow not available")
    from p2pfl.learning.frameworks.tensorflow.callbacks.scaffold_callback import SCAFFOLDCallback

    callback = SCAFFOLDCallback()
    assert callback.K == 0
    callback.on_train_batch_end(batch=0)
    assert callback.K == 1
    callback.on_train_batch_end(batch=1)
    assert callback.K == 2


def test_scaffold_tf_on_train_end_computes_delta_c_correctly():
    """Test TF on_train_end computes delta_c_i and delta_y_i following SCAFFOLD formula."""
    tf = pytest.importorskip("tensorflow", reason="TensorFlow not available")
    np = pytest.importorskip("numpy", reason="NumPy not available")
    from p2pfl.learning.frameworks.tensorflow.callbacks.scaffold_callback import SCAFFOLDCallback

    callback = SCAFFOLDCallback()
    callback.saved_lr = 0.1
    callback.K = 5

    # c_i starts at zeros
    callback.c_i = [tf.Variable([0.0, 0.0])]
    # initial model params (x_g)
    callback.initial_model_params = [tf.Variable([1.0, 2.0])]

    # Current model params (y_i) after training: [0.5, 1.5]
    y_param = tf.Variable([0.5, 1.5])
    mock_model = MagicMock()
    mock_model.trainable_variables = [y_param]
    callback.set_model(mock_model)

    callback.on_train_end()

    # adjustment = (x_g - y_i) / (K * lr) = ([1.0, 2.0] - [0.5, 1.5]) / (5 * 0.1) = [0.5, 0.5] / 0.5 = [1.0, 1.0]
    # new_c_i = [0.0, 0.0] + [1.0, 1.0] = [1.0, 1.0]
    # delta_c_i = [1.0, 1.0] - [0.0, 0.0] = [1.0, 1.0]
    assert "delta_c_i" in callback.additional_info
    assert "delta_y_i" in callback.additional_info

    delta_c_i = callback.additional_info["delta_c_i"]
    np.testing.assert_allclose(delta_c_i[0], [1.0, 1.0])

    # delta_y_i = y_i - x_g = [0.5, 1.5] - [1.0, 2.0] = [-0.5, -0.5]
    delta_y_i = callback.additional_info["delta_y_i"]
    np.testing.assert_allclose(delta_y_i[0], [-0.5, -0.5])


def test_scaffold_tf_on_train_end_updates_c_i():
    """Test TF on_train_end correctly updates c_i in place via assign_add."""
    tf = pytest.importorskip("tensorflow", reason="TensorFlow not available")
    np = pytest.importorskip("numpy", reason="NumPy not available")
    from p2pfl.learning.frameworks.tensorflow.callbacks.scaffold_callback import SCAFFOLDCallback

    callback = SCAFFOLDCallback()
    callback.saved_lr = 0.5
    callback.K = 2

    # c_i starts at [0.3, 0.7]
    callback.c_i = [tf.Variable([0.3, 0.7])]
    callback.initial_model_params = [tf.Variable([2.0, 4.0])]

    y_param = tf.Variable([1.0, 3.0])
    mock_model = MagicMock()
    mock_model.trainable_variables = [y_param]
    callback.set_model(mock_model)

    callback.on_train_end()

    # adjustment = ([2.0, 4.0] - [1.0, 3.0]) / (2 * 0.5) = [1.0, 1.0] / 1.0 = [1.0, 1.0]
    # new_c_i = [0.3, 0.7] + [1.0, 1.0] = [1.3, 1.7]
    np.testing.assert_allclose(callback.c_i[0].numpy(), [1.3, 1.7])


def test_scaffold_tf_on_train_end_raises_without_init():
    """Test TF on_train_end raises if attributes are not initialized."""
    tf = pytest.importorskip("tensorflow", reason="TensorFlow not available")
    from p2pfl.learning.frameworks.tensorflow.callbacks.scaffold_callback import SCAFFOLDCallback

    callback = SCAFFOLDCallback()
    mock_model = MagicMock()
    mock_model.trainable_variables = [tf.Variable([1.0])]
    callback.set_model(mock_model)

    with pytest.raises(AttributeError, match="Necessary attributes are not initialized"):
        callback.on_train_end()


def test_scaffold_tf_on_train_end_multiple_params():
    """Test TF SCAFFOLD delta computation with multiple parameter groups."""
    tf = pytest.importorskip("tensorflow", reason="TensorFlow not available")
    np = pytest.importorskip("numpy", reason="NumPy not available")
    from p2pfl.learning.frameworks.tensorflow.callbacks.scaffold_callback import SCAFFOLDCallback

    callback = SCAFFOLDCallback()
    callback.saved_lr = 0.01
    callback.K = 10

    callback.c_i = [tf.Variable([0.0]), tf.Variable([0.0, 0.0])]
    callback.initial_model_params = [tf.Variable([1.0]), tf.Variable([2.0, 3.0])]

    y1 = tf.Variable([0.9])
    y2 = tf.Variable([1.8, 2.7])
    mock_model = MagicMock()
    mock_model.trainable_variables = [y1, y2]
    callback.set_model(mock_model)

    callback.on_train_end()

    # param 0: adjustment = (1.0 - 0.9) / (10 * 0.01) = 0.1 / 0.1 = 1.0
    # param 1: adjustment = ([2.0, 3.0] - [1.8, 2.7]) / 0.1 = [2.0, 3.0]
    delta_c_i = callback.additional_info["delta_c_i"]
    np.testing.assert_allclose(delta_c_i[0], [1.0], atol=1e-6)
    np.testing.assert_allclose(delta_c_i[1], [2.0, 3.0], atol=1e-6)

    delta_y_i = callback.additional_info["delta_y_i"]
    np.testing.assert_allclose(delta_y_i[0], [-0.1], atol=1e-6)
    np.testing.assert_allclose(delta_y_i[1], [-0.2, -0.3], atol=1e-6)
