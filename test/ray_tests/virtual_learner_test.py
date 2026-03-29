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

These tests mock the SuperActorPool to avoid actually running Ray actors,
which would fail with MagicMock learners (not serializable).
"""

import contextlib
from unittest.mock import MagicMock, patch

import pytest

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel

# These tests DON'T need Ray - they mock SuperActorPool entirely

with contextlib.suppress(ImportError):
    from p2pfl.learning.frameworks.simulation.actor_pool import SuperActorPool
    from p2pfl.learning.frameworks.simulation.virtual_learner import VirtualNodeLearner


@pytest.fixture
def mock_actor_pool():
    """Mock SuperActorPool to avoid Ray initialization."""
    mock_pool = MagicMock(spec=SuperActorPool)
    with patch(
        "p2pfl.learning.frameworks.simulation.virtual_learner.SuperActorPool",
        return_value=mock_pool,
    ):
        yield mock_pool


def test_virtual_node_learner_initialization(mock_actor_pool):
    """Test the initialization of the VirtualNodeLearner class."""
    learner = MagicMock(spec=Learner)
    addr = "test_addr"
    virtual_learner = VirtualNodeLearner(learner)
    virtual_learner.set_addr(addr)
    assert virtual_learner.learner == learner
    assert virtual_learner.actor_pool == mock_actor_pool
    assert virtual_learner.addr == addr


def test_set_model(mock_actor_pool):
    """Test the set_model method of the VirtualNodeLearner class."""
    learner = MagicMock(spec=Learner)
    virtual_learner = VirtualNodeLearner(learner)
    virtual_learner.set_addr("test_addr")
    model = MagicMock(spec=P2PFLModel)
    virtual_learner.set_model(model)
    learner.set_model.assert_called_once_with(model)


def test_get_model(mock_actor_pool):
    """Test the get_model method of the VirtualNodeLearner class."""
    learner = MagicMock(spec=Learner)
    virtual_learner = VirtualNodeLearner(learner)
    virtual_learner.set_addr("test_addr")
    model = MagicMock(spec=P2PFLModel)
    learner.get_model.return_value = model
    result = virtual_learner.get_model()
    assert result == model
    learner.get_model.assert_called_once()


def test_set_data(mock_actor_pool):
    """Test the set_data method of the VirtualNodeLearner class."""
    learner = MagicMock(spec=Learner)
    virtual_learner = VirtualNodeLearner(learner)
    virtual_learner.set_addr("test_addr")
    data = MagicMock(spec=P2PFLDataset)
    virtual_learner.set_data(data)
    learner.set_data.assert_called_once_with(data)


def test_get_data(mock_actor_pool):
    """Test the get_data method of the VirtualNodeLearner class."""
    learner = MagicMock(spec=Learner)
    virtual_learner = VirtualNodeLearner(learner)
    virtual_learner.set_addr("test_addr")
    data = MagicMock(spec=P2PFLDataset)
    learner.get_data.return_value = data
    result = virtual_learner.get_data()
    assert result == data
    learner.get_data.assert_called_once()


def test_set_epochs(mock_actor_pool):
    """Test the set_epochs method of the VirtualNodeLearner class."""
    learner = MagicMock(spec=Learner)
    virtual_learner = VirtualNodeLearner(learner)
    virtual_learner.set_addr("test_addr")
    epochs = 10
    virtual_learner.set_epochs(epochs)
    learner.set_epochs.assert_called_once_with(epochs)


def test_fit(mock_actor_pool):
    """Test the fit method of the VirtualNodeLearner class."""
    learner = MagicMock(spec=Learner)
    virtual_learner = VirtualNodeLearner(learner)
    virtual_learner.set_addr("test_addr")
    model = MagicMock(spec=P2PFLModel)

    # Configure the mock to return expected values
    mock_actor_pool.get_learner_result.return_value = ("test_addr", model)

    result = virtual_learner.fit()

    mock_actor_pool.submit_learner_job.assert_called_once()
    mock_actor_pool.get_learner_result.assert_called_once_with("test_addr", None)
    learner.set_model.assert_called_once_with(model)
    assert result == model


def _test_interrupt_fit():
    """Test the interrupt_fit method of the VirtualNodeLearner class."""
    # TODO: Not implemented in VirtualNodeLearner
    pass


def test_evaluate(mock_actor_pool):
    """Test the evaluate method of the VirtualNodeLearner class."""
    learner = MagicMock(spec=Learner)
    virtual_learner = VirtualNodeLearner(learner)
    virtual_learner.set_addr("test_addr")
    evaluation_result = {"accuracy": 0.9}

    # Configure the mock to return expected values
    mock_actor_pool.get_learner_result.return_value = ("test_addr", evaluation_result)

    result = virtual_learner.evaluate()

    mock_actor_pool.submit_learner_job.assert_called_once()
    mock_actor_pool.get_learner_result.assert_called_once_with("test_addr", None)
    assert result == evaluation_result
