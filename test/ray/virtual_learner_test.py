#
# This file is part of the p2pfl distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2025 Pedro Guijas Bravo.
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
Virtual node tests.

These tests mock the Ray actor and placement group to avoid actually running Ray,
which would fail with MagicMock learners (not serializable).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    import ray  # noqa: F401

    RAY_INSTALLED = True
except ImportError:
    RAY_INSTALLED = False

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel

if RAY_INSTALLED:
    from p2pfl.learning.frameworks.ray.virtual_learner import VirtualNodeLearner

# Skip all tests in this module if Ray is not installed
pytestmark = pytest.mark.skipif(not RAY_INSTALLED, reason="Ray is not installed")


def create_mock_learner(address: str = "") -> MagicMock:
    """Create a mock Learner with the address attribute set (required by VirtualNodeLearner)."""
    learner = MagicMock(spec=Learner)
    learner.address = address  # NodeComponent sets this via metaclass; mocks need it explicitly
    return learner


@pytest.fixture
def mock_ray_components():
    """Mock VirtualLearnerActor and PlacementGroupManager to avoid Ray initialization."""
    mock_actor = MagicMock()
    mock_actor_options = MagicMock()
    mock_actor_options.remote.return_value = mock_actor

    mock_actor_class = MagicMock()
    mock_actor_class.options.return_value = mock_actor_options

    mock_pg_manager = MagicMock()
    mock_pg_manager.get_placement_group.return_value = None

    with (
        patch(
            "p2pfl.learning.frameworks.ray.virtual_learner.VirtualLearnerActor",
            mock_actor_class,
        ),
        patch(
            "p2pfl.learning.frameworks.ray.virtual_learner.PlacementGroupManager",
            return_value=mock_pg_manager,
        ),
        patch("p2pfl.learning.frameworks.ray.virtual_learner.ray") as mock_ray,
    ):
        # Setup ray.get to return whatever the remote call returns
        mock_ray.get.side_effect = lambda x: x
        yield {
            "actor": mock_actor,
            "actor_class": mock_actor_class,
            "pg_manager": mock_pg_manager,
            "ray": mock_ray,
        }


def test_virtual_node_learner_initialization(mock_ray_components):
    """Test the initialization of the VirtualNodeLearner class."""
    learner = create_mock_learner()
    virtual_learner = VirtualNodeLearner(learner)
    address = "test_addr"
    virtual_learner.set_address(address)
    assert virtual_learner.address == address
    # Verify actor was created
    mock_ray_components["actor_class"].options.assert_called_once()


def test_set_model(mock_ray_components):
    """Test the set_model method of the VirtualNodeLearner class."""
    learner = create_mock_learner()
    virtual_learner = VirtualNodeLearner(learner)
    virtual_learner.set_address("test_addr")
    model = MagicMock(spec=P2PFLModel)

    virtual_learner.set_model(model)

    # Verify set_model was called on the actor
    mock_ray_components["actor"].set_model.remote.assert_called_once_with(model)


def test_get_model(mock_ray_components):
    """Test the get_model method of the VirtualNodeLearner class."""
    learner = create_mock_learner()
    virtual_learner = VirtualNodeLearner(learner)
    virtual_learner.set_address("test_addr")
    model = MagicMock(spec=P2PFLModel)

    mock_ray_components["actor"].get_model.remote.return_value = model

    result = virtual_learner.get_model()

    mock_ray_components["actor"].get_model.remote.assert_called_once()
    assert result == model


def test_set_data(mock_ray_components):
    """Test the set_data method of the VirtualNodeLearner class."""
    learner = create_mock_learner()
    virtual_learner = VirtualNodeLearner(learner)
    virtual_learner.set_address("test_addr")
    data = MagicMock(spec=P2PFLDataset)

    virtual_learner.set_data(data)

    mock_ray_components["actor"].set_data.remote.assert_called_once_with(data)


def test_get_data(mock_ray_components):
    """Test the get_data method of the VirtualNodeLearner class."""
    learner = create_mock_learner()
    virtual_learner = VirtualNodeLearner(learner)
    virtual_learner.set_address("test_addr")
    data = MagicMock(spec=P2PFLDataset)

    mock_ray_components["actor"].get_data.remote.return_value = data

    result = virtual_learner.get_data()

    mock_ray_components["actor"].get_data.remote.assert_called_once()
    assert result == data


def test_set_epochs(mock_ray_components):
    """Test the set_epochs method of the VirtualNodeLearner class."""
    learner = create_mock_learner()
    virtual_learner = VirtualNodeLearner(learner)
    virtual_learner.set_address("test_addr")
    epochs = 10

    virtual_learner.set_epochs(epochs)

    mock_ray_components["actor"].set_epochs.remote.assert_called_once_with(epochs)


@pytest.mark.asyncio
async def test_fit(mock_ray_components):
    """Test the fit method of the VirtualNodeLearner class."""
    learner = create_mock_learner()
    virtual_learner = VirtualNodeLearner(learner)
    virtual_learner.set_address("test_addr")

    # Mock the async remote call
    mock_ray_components["actor"].fit.remote = AsyncMock()

    await virtual_learner.fit()

    mock_ray_components["actor"].fit.remote.assert_called_once()


def _test_interrupt_fit():
    """Test the interrupt_fit method of the VirtualNodeLearner class."""
    # TODO: Not implemented in VirtualNodeLearner
    pass


@pytest.mark.asyncio
async def test_evaluate(mock_ray_components):
    """Test the evaluate method of the VirtualNodeLearner class."""
    learner = create_mock_learner()
    virtual_learner = VirtualNodeLearner(learner)
    virtual_learner.set_address("test_addr")
    evaluation_result = {"accuracy": 0.9}

    # Mock the async remote call
    mock_ray_components["actor"].evaluate.remote = AsyncMock(return_value=evaluation_result)

    result = await virtual_learner.evaluate()

    mock_ray_components["actor"].evaluate.remote.assert_called_once()
    assert result == evaluation_result
