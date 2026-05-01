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
"""Weight-based framework tests (PyTorch, TensorFlow, Flax)."""

import contextlib
from collections.abc import Generator
from unittest.mock import MagicMock

import numpy as np
import pytest
from datasets import DatasetDict, load_dataset  # type: ignore

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks import Framework
from p2pfl.learning.frameworks.exceptions import ModelNotMatchingError
from p2pfl.learning.frameworks.learner_factory import LearnerFactory
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.workflow.engine.experiment import Experiment

with contextlib.suppress(ImportError):
    import tensorflow as tf

    from p2pfl.examples.mnist.model.mlp_tensorflow import model_build_fn as model_build_fn_tensorflow
    from p2pfl.learning.frameworks.tensorflow.keras_dataset import KerasExportStrategy
    from p2pfl.learning.frameworks.tensorflow.keras_model import KerasModel

with contextlib.suppress(ImportError):
    import jax
    import jax.numpy as jnp

    from p2pfl.examples.mnist.model.mlp_flax import MLP as MLP_FLASK
    from p2pfl.learning.frameworks.flax.flax_dataset import FlaxExportStrategy
    from p2pfl.learning.frameworks.flax.flax_model import FlaxModel

with contextlib.suppress(ImportError):
    import torch
    from torch.utils.data import DataLoader

    from p2pfl.examples.mnist.model.mlp_pytorch import model_build_fn as model_build_fn_torch
    from p2pfl.learning.frameworks.pytorch.lightning_dataset import PyTorchExportStrategy, TorchvisionDatasetFactory
    from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel

############################
#    PyTorch Model Tests
############################


def test_get_set_params_torch():
    """Test setting and getting parameters."""
    # Create the model
    p2pfl_model = model_build_fn_torch()
    # Modify parameters
    params = p2pfl_model.get_parameters()
    params_og = [layer.copy() for layer in p2pfl_model.get_parameters()]
    for i, layer in enumerate(params):
        params[i] = layer + 1
    # Set parameters
    p2pfl_model.set_parameters(params)
    # Check if the parameters are different (+1)
    for layer_og, layer_new in zip(params_og, p2pfl_model.get_parameters(), strict=False):
        assert np.all(layer_og + 1 == layer_new)


def test_encoding_torch():
    """Test encoding and decoding of parameters."""
    p2pfl_model1 = model_build_fn_torch()
    encoded_params = p2pfl_model1.encode_parameters()

    p2pfl_model2 = model_build_fn_torch()
    decoded_params, additional_info = p2pfl_model2.decode_parameters(encoded_params)
    p2pfl_model2.set_parameters(decoded_params)
    p2pfl_model2.additional_info = additional_info

    assert encoded_params == p2pfl_model1.encode_parameters()
    assert additional_info == p2pfl_model1.additional_info


def test_wrong_encoding_torch():
    """Test wrong encoding of parameters."""
    p2pfl_model1 = model_build_fn_torch()
    encoded_params = p2pfl_model1.encode_parameters()
    mobile_net = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=False)
    p2pfl_model2 = LightningModel(mobile_net)
    decoded_params, _ = p2pfl_model2.decode_parameters(encoded_params)
    # Check that raises
    with pytest.raises(ModelNotMatchingError):
        p2pfl_model2.set_parameters(decoded_params)


###############################
#    TensorFlow Model Tests
###############################


def test_get_set_params_tensorflow():
    """Test setting and getting parameters."""
    # Create the model
    p2pfl_model = model_build_fn_tensorflow()
    # Modify parameters
    params = p2pfl_model.get_parameters()
    params_og = [layer.copy() for layer in p2pfl_model.get_parameters()]
    for i, layer in enumerate(params):
        params[i] = layer + 1
    # Set parameters
    p2pfl_model.set_parameters(params)
    # Check if the parameters are different (+1)
    for layer_og, layer_new in zip(params_og, p2pfl_model.get_parameters(), strict=False):
        assert np.all(layer_og + 1 == layer_new)


def test_encoding_tensorflow():
    """Test encoding and decoding of parameters."""
    p2pfl_model1 = model_build_fn_tensorflow()
    encoded_params = p2pfl_model1.encode_parameters()
    p2pfl_model2 = model_build_fn_tensorflow()
    decoded_params, additional_info = p2pfl_model2.decode_parameters(encoded_params)
    p2pfl_model2.set_parameters(decoded_params)

    assert encoded_params == p2pfl_model1.encode_parameters()
    assert additional_info == p2pfl_model1.additional_info


def test_wrong_encoding_tensorflow():
    """Test wrong encoding of parameters."""
    p2pfl_model1 = model_build_fn_tensorflow()
    encoded_params = p2pfl_model1.encode_parameters()
    mobile_net = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    p2pfl_model2 = KerasModel(mobile_net)
    decoded_params = p2pfl_model2.decode_parameters(encoded_params)
    # Check that raises
    with pytest.raises(ModelNotMatchingError):
        p2pfl_model2.set_parameters(decoded_params)


####################################
#    Flax Model Tests (disabled)
####################################


@pytest.mark.skip(reason="Flax support disabled pending JAX integration")
def test_get_set_params_flax():
    """Test setting and getting parameters."""
    # Create the model
    model = MLP_FLASK()
    seed = jax.random.PRNGKey(0)
    model_params = model.init(seed, jnp.ones((1, 28, 28)))["params"]
    p2pfl_model = FlaxModel(model, model_params)

    # Save internal flax-model repr
    _flax_params = p2pfl_model.model_params.copy()
    params = p2pfl_model.get_parameters()
    p2pfl_model.set_parameters(params)

    # Check that flax to numpy arrays transformation works
    for layer in _flax_params:
        for param in _flax_params[layer]:
            assert np.array_equal(
                _flax_params[layer][param], p2pfl_model.model_params[layer][param]
            ), f"Mismatch found in {layer} - {param}"

    # Modify parameters
    params = p2pfl_model.get_parameters()
    params_og = [layer.copy() for layer in p2pfl_model.get_parameters()]
    for i, layer in enumerate(params):
        params[i] = layer + 1
    # Set parameters
    p2pfl_model.set_parameters(params)
    # Check if the parameters are different (+1)
    for layer_og, layer_new in zip(params_og, p2pfl_model.get_parameters(), strict=False):
        assert np.all(layer_og + 1 == layer_new)


@pytest.mark.skip(reason="Flax support disabled pending JAX integration")
def test_encoding_flax():
    """Test encoding and decoding of parameters."""
    model1 = MLP_FLASK()
    seed = jax.random.PRNGKey(0)
    model_params = model1.init(seed, jnp.ones((1, 28, 28)))["params"]
    p2pfl_model1 = FlaxModel(model=model1, init_params=model_params)
    encoded_params = p2pfl_model1.encode_parameters()

    model2 = MLP_FLASK()
    seed = jax.random.PRNGKey(1)
    model_params = model2.init(seed, jnp.ones((1, 28, 28)))["params"]
    p2pfl_model2 = FlaxModel(model2, init_params=model_params)
    decoded_params, additional_info = p2pfl_model2.decode_parameters(encoded_params)
    p2pfl_model2.set_parameters(decoded_params)

    for arr1, arr2 in zip(p2pfl_model1.get_parameters(), p2pfl_model2.get_parameters(), strict=False):
        assert np.array_equal(arr1, arr2)
        assert p2pfl_model1.additional_info == p2pfl_model2.additional_info


@pytest.mark.skip(reason="Flax support disabled pending JAX integration")
def test_wrong_encoding_flax():
    """Test wrong encoding of parameters."""
    model1 = MLP_FLASK()
    seed = jax.random.PRNGKey(0)
    model_params1 = model1.init(seed, jnp.ones((1, 28, 28)))["params"]
    p2pfl_model1 = FlaxModel(model=model1, init_params=model_params1)
    encoded_params = p2pfl_model1.encode_parameters()
    model2 = MLP_FLASK()
    model2.hidden_sizes = (256, 128, 256, 128)
    model_params2 = model2.init(seed, jnp.ones((1, 28, 28)))["params"]
    p2pfl_model2 = FlaxModel(model=model2, init_params=model_params2)
    decoded_params = p2pfl_model1.decode_parameters(encoded_params)
    # Check that raises
    with pytest.raises(ModelNotMatchingError):
        p2pfl_model2.set_parameters(decoded_params)


###########################
#    PyTorch Data Tests
###########################


def test_torchvision_dataset_factory_mnist():
    """Test the TorchvisionDatasetFactory for MNIST."""
    train_dataset = TorchvisionDatasetFactory.get_mnist(cache_dir=".", train=True, download=True)
    test_dataset = TorchvisionDatasetFactory.get_mnist(cache_dir=".", train=False, download=True)

    assert isinstance(train_dataset, P2PFLDataset)
    assert isinstance(test_dataset, P2PFLDataset)

    assert train_dataset.get_num_samples() > 0
    assert test_dataset.get_num_samples() > 0

    # Check if the data is loaded correctly
    sample = train_dataset.get(0)
    assert "image" in sample
    assert "label" in sample

    # Check if the data is loaded correctly
    assert sample["image"].size == (28, 28)


def test_pytorch_export_strategy():
    """Test the PyTorchExportStrategy."""
    dataset = TorchvisionDatasetFactory.get_mnist(cache_dir=".", train=True, download=True)
    dataset.set_batch_size(1)

    export_strategy = PyTorchExportStrategy()
    train_dataloader = dataset.export(export_strategy, train_loader=True)
    test_dataloader = dataset.export(export_strategy, train_loader=False)

    assert isinstance(train_dataloader, DataLoader)
    assert isinstance(test_dataloader, DataLoader)

    # Check if data
    assert len(train_dataloader) > 0
    assert len(test_dataloader) > 0

    # Check if the data is loaded correctly
    sample = next(iter(train_dataloader))
    assert "image" in sample
    assert "label" in sample

    # Check if the data is loaded correctly
    # datasets 4+ adds channel dimension: (B, C, H, W) = (1, 1, 28, 28)
    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].size() == (1, 1, 28, 28)


##############################
#    TensorFlow Data Tests
##############################


def test_tensorflow_export_strategy():
    """Test the KerasExportStrategy."""
    dataset = TorchvisionDatasetFactory.get_mnist(cache_dir=".", train=True, download=True)
    dataset.set_batch_size(1)

    export_strategy = KerasExportStrategy()
    train_data = dataset.export(export_strategy, train_loader=True)
    test_data = dataset.export(export_strategy, train_loader=False)

    assert isinstance(train_data, tf.data.Dataset)
    assert isinstance(test_data, tf.data.Dataset)

    # Check if data
    assert len(train_data) > 0
    assert len(test_data) > 0

    # Check if the data is loaded correctly
    sample = next(iter(train_data))
    assert isinstance(sample, tuple)

    # Check if the data is loaded correctly
    assert isinstance(sample[0], tf.Tensor)
    assert sample[0].shape == (1, 28, 28)


###################################
#    Flax Data Tests (disabled)
###################################


@pytest.mark.skip(reason="Flax support disabled pending JAX integration")
def test_flax_export_strategy():
    """Test the FlaxExportStrategy."""
    dataset = TorchvisionDatasetFactory.get_mnist(cache_dir=".", train=True, download=True)
    dataset.set_batch_size(1)

    export_strategy = FlaxExportStrategy()
    train_data = dataset.export(export_strategy, train_loader=True)
    test_data = dataset.export(export_strategy, train_loader=False)

    assert isinstance(train_data, Generator)
    assert isinstance(test_data, Generator)

    # Check if data
    assert train_data is not None
    assert test_data is not None

    # Check if the data is loaded correctly
    x, y = next(iter(train_data))

    assert isinstance(x, jnp.ndarray)
    assert x.shape == (1, 28, 28)

    assert isinstance(y, jnp.ndarray)
    assert y.shape == (1,)


################################
#    Learner Training Tests
################################


@pytest.mark.asyncio
@pytest.mark.parametrize("build_model_fn", [model_build_fn_torch, model_build_fn_tensorflow])  # TODO: Flax
async def test_learner_train(build_model_fn):
    """Test the training and testing of the learner."""
    # Dataset
    dataset = P2PFLDataset(
        DatasetDict(
            {
                "train": load_dataset("p2pfl/MNIST", split="train[:100]"),
                "test": load_dataset("p2pfl/MNIST", split="test[:10]"),
            }
        )
    )

    # Create the model
    p2pfl_model = build_model_fn()

    # Dont care about the seed
    Settings.general.SEED = None

    node_name = "unknown-node"
    with contextlib.suppress(Exception):
        logger.register_node(node_name)
    experiment = Experiment(exp_name="test_experiment-torch", total_rounds=1)
    logger.experiment_started(node_name, experiment)
    # Learner
    learner = LearnerFactory.create_learner(p2pfl_model)()
    learner.set_address(node_name)
    learner.set_model(p2pfl_model)
    learner.set_data(dataset)

    # Train
    learner.set_epochs(1)
    await learner.fit()

    # Test
    await learner.evaluate()


###################################
#    LearnerFactory Tests
###################################


def test_learner_factory_pytorch():
    """LearnerFactory returns LightningLearner for a PyTorch model."""
    from p2pfl.learning.frameworks.pytorch.lightning_learner import LightningLearner

    model = model_build_fn_torch()
    learner_cls = LearnerFactory.create_learner(model)
    assert learner_cls is LightningLearner


def test_learner_factory_tensorflow():
    """LearnerFactory returns KerasLearner for a TensorFlow model."""
    from p2pfl.learning.frameworks.tensorflow.keras_learner import KerasLearner

    model = model_build_fn_tensorflow()
    learner_cls = LearnerFactory.create_learner(model)
    assert learner_cls is KerasLearner


def test_learner_factory_unsupported_framework():
    """LearnerFactory raises ValueError for an unknown framework string."""
    mock_model = MagicMock()
    mock_model.get_framework.return_value = "unknown_framework_xyz"
    with pytest.raises(ValueError, match="Unsupported framework"):
        LearnerFactory.create_learner(mock_model)


def test_learner_factory_flax():
    """LearnerFactory returns FlaxLearner for a Flax model."""
    pytest.importorskip("jax")
    from p2pfl.learning.frameworks.flax.flax_learner import FlaxLearner

    mock_model = MagicMock()
    mock_model.get_framework.return_value = Framework.FLAX.value
    learner_cls = LearnerFactory.create_learner(mock_model)
    assert learner_cls is FlaxLearner


def test_learner_factory_xgboost():
    """LearnerFactory returns XGBoostLearner for an XGBoost model."""
    pytest.importorskip("xgboost")
    from p2pfl.learning.frameworks.xgboost.xgboost_learner import XGBoostLearner

    mock_model = MagicMock()
    mock_model.get_framework.return_value = Framework.XGBOOST.value
    learner_cls = LearnerFactory.create_learner(mock_model)
    assert learner_cls is XGBoostLearner


###################################
#    Learner Base Tests
###################################


class TestLearnerBase:
    """Tests for base Learner error paths and accessor methods."""

    def _make_learner(self):
        """Create a LightningLearner with no model/data (for testing accessors)."""
        from p2pfl.learning.frameworks.pytorch.lightning_learner import LightningLearner

        learner = LightningLearner()
        return learner

    def test_get_model_without_set_raises(self):
        """get_model raises ValueError when model is not set."""
        learner = self._make_learner()
        with pytest.raises(ValueError, match="Model not initialized"):
            learner.get_model()

    def test_get_data_without_set_raises(self):
        """get_data raises ValueError when data is not set."""
        learner = self._make_learner()
        with pytest.raises(ValueError, match="Data not initialized"):
            learner.get_data()

    def test_epochs_accessors(self):
        """get/set_epochs round-trips correctly."""
        learner = self._make_learner()
        assert learner.get_epochs() == 1
        learner.set_epochs(5)
        assert learner.get_epochs() == 5

    def test_steps_per_epoch_accessors(self):
        """get/set_steps_per_epoch round-trips correctly."""
        learner = self._make_learner()
        assert learner.get_steps_per_epoch() is None
        learner.set_steps_per_epoch(100)
        assert learner.get_steps_per_epoch() == 100

    def test_init_with_aggregator(self):
        """Learner.__init__ with an aggregator calls indicate_aggregator."""
        from p2pfl.learning.frameworks.pytorch.lightning_learner import LightningLearner

        mock_agg = MagicMock()
        mock_agg.get_required_callbacks.return_value = []
        learner = LightningLearner()
        learner.set_address("test-addr")
        learner.indicate_aggregator(mock_agg)
        # Should not raise and callbacks may be empty since aggregator has no required callbacks
        assert isinstance(learner.callbacks, list)


###################################
#    LearnerDecorator Tests
###################################


class TestLearnerDecorator:
    """Tests for the LearnerDecorator delegation."""

    def _make_decorated(self):
        from p2pfl.learning.frameworks.learner import LearnerDecorator
        from p2pfl.learning.frameworks.pytorch.lightning_learner import LightningLearner

        inner = LightningLearner()
        inner.set_address("test-addr")
        dec = LearnerDecorator(inner)
        # The decorator's own address must also be set for the metaclass check
        dec.address = "test-addr"
        return dec, inner

    def test_set_address_delegates(self):
        """set_address delegates to inner learner."""
        dec, inner = self._make_decorated()
        result = dec.set_address("new-addr")
        assert result == "new-addr"
        assert inner.address == "new-addr"

    def test_model_delegation(self):
        """set_model/get_model delegates to inner learner."""
        dec, inner = self._make_decorated()
        model = model_build_fn_torch()
        dec.set_model(model)
        assert dec.get_model() is inner.get_model()

    def test_data_delegation(self):
        """set_data/get_data delegates to inner learner."""
        dec, inner = self._make_decorated()
        data = P2PFLDataset(
            DatasetDict(
                {
                    "train": load_dataset("p2pfl/MNIST", split="train[:10]"),
                    "test": load_dataset("p2pfl/MNIST", split="test[:5]"),
                }
            )
        )
        dec.set_data(data)
        assert dec.get_data() is inner.get_data()

    def test_epochs_delegation(self):
        """set/get_epochs delegates to inner learner."""
        dec, inner = self._make_decorated()
        dec.set_epochs(10)
        assert dec.get_epochs() == 10
        assert inner.get_epochs() == 10

    def test_steps_per_epoch_delegation(self):
        """set/get_steps_per_epoch delegates to inner learner."""
        dec, inner = self._make_decorated()
        dec.set_steps_per_epoch(50)
        assert dec.get_steps_per_epoch() == 50

    def test_indicate_aggregator_delegation(self):
        """indicate_aggregator delegates to inner learner."""
        dec, inner = self._make_decorated()
        mock_agg = MagicMock()
        mock_agg.get_required_callbacks.return_value = []
        dec.indicate_aggregator(mock_agg)

    def test_get_framework_delegation(self):
        """get_framework delegates to inner learner."""
        dec, inner = self._make_decorated()
        assert dec.get_framework() == inner.get_framework()

    def test_update_callbacks_delegation(self):
        """update_callbacks_with_model_info delegates to inner learner."""
        dec, inner = self._make_decorated()
        model = model_build_fn_torch()
        dec.set_model(model)
        # Should not raise
        dec.update_callbacks_with_model_info()

    def test_add_callback_info_delegation(self):
        """add_callback_info_to_model delegates to inner learner."""
        dec, inner = self._make_decorated()
        model = model_build_fn_torch()
        dec.set_model(model)
        dec.add_callback_info_to_model()

    @pytest.mark.asyncio
    async def test_fit_delegation(self):
        """Fit delegates to inner learner and returns a model."""
        dec, inner = self._make_decorated()
        model = model_build_fn_torch()
        dec.set_model(model)
        data = P2PFLDataset(
            DatasetDict(
                {
                    "train": load_dataset("p2pfl/MNIST", split="train[:10]"),
                    "test": load_dataset("p2pfl/MNIST", split="test[:5]"),
                }
            )
        )
        dec.set_data(data)
        dec.set_epochs(1)
        Settings.general.SEED = None
        result = await dec.fit()
        assert result is not None

    @pytest.mark.asyncio
    async def test_evaluate_delegation(self):
        """Evaluate delegates to inner learner."""
        dec, inner = self._make_decorated()
        model = model_build_fn_torch()
        dec.set_model(model)
        data = P2PFLDataset(
            DatasetDict(
                {
                    "train": load_dataset("p2pfl/MNIST", split="train[:10]"),
                    "test": load_dataset("p2pfl/MNIST", split="test[:5]"),
                }
            )
        )
        dec.set_data(data)
        result = await dec.evaluate()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_train_on_batch_delegation(self):
        """Train_on_batch delegates to inner learner (raises NotImplementedError for PyTorch)."""
        dec, inner = self._make_decorated()
        model = model_build_fn_torch()
        dec.set_model(model)
        data = P2PFLDataset(
            DatasetDict(
                {
                    "train": load_dataset("p2pfl/MNIST", split="train[:10]"),
                    "test": load_dataset("p2pfl/MNIST", split="test[:5]"),
                }
            )
        )
        dec.set_data(data)
        # PyTorch Lightning does not support batch training
        with pytest.raises(NotImplementedError):
            await dec.train_on_batch()

    @pytest.mark.asyncio
    async def test_interrupt_fit_delegation(self):
        """Interrupt_fit delegates to inner learner."""
        dec, inner = self._make_decorated()
        # Should not raise even without ongoing fit
        await dec.interrupt_fit()
