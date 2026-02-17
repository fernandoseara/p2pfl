#
# This file is part of the federated_learning_p2p (p2pfl) distribution
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

"""Tests for AsyncDFL."""

from unittest.mock import MagicMock, patch

import pytest

from p2pfl.workflow.async_dfl.workflow import (
    AsyncDFL,
    AsyncPeerState,
    get_states,
    get_transitions,
)
from p2pfl.workflow.engine.experiment import Experiment
from p2pfl.workflow.factory import WorkflowType


@pytest.fixture
def mock_node():
    """Create a mock node for async workflow testing."""
    node = MagicMock()
    node.address = "test_node_address"

    # Mock communication protocol
    comm_protocol = MagicMock()
    node.get_communication_protocol.return_value = comm_protocol
    node.communication_protocol = comm_protocol

    # Mock learner
    learner = MagicMock()
    model = MagicMock()
    learner.get_model.return_value = model
    node.get_learner.return_value = learner

    # Set workflow_type property
    node.workflow_type = WorkflowType.ASYNC

    return node


class TestAsyncWorkflowStatesAndTransitions:
    """Tests for async workflow state machine configuration."""

    def test_get_states_returns_list(self):
        """Test that get_states returns a list of state dicts."""
        states = get_states()
        assert isinstance(states, list)
        assert len(states) > 0
        assert all(isinstance(s, dict) for s in states)

    def test_get_transitions_returns_list(self):
        """Test that get_transitions returns a list of transition dicts."""
        transitions = get_transitions()
        assert isinstance(transitions, list)
        assert len(transitions) > 0
        assert all(isinstance(t, dict) for t in transitions)

    def test_states_have_required_keys(self):
        """Test that states have required keys."""
        states = get_states()
        for state in states:
            assert "name" in state

    def test_transitions_have_required_keys(self):
        """Test that transitions have required keys."""
        transitions = get_transitions()
        for transition in transitions:
            assert "trigger" in transition
            assert "source" in transition
            assert "dest" in transition

    def test_initial_state_exists(self):
        """Test that waitingSetup state exists."""
        states = get_states()
        state_names = [s["name"] for s in states]
        assert "waitingSetup" in state_names

    def test_final_state_exists(self):
        """Test that trainingFinished state exists."""
        states = get_states()
        state_names = [s["name"] for s in states]
        assert "trainingFinished" in state_names

    def test_training_round_state_exists(self):
        """Test that trainingRound compound state exists."""
        states = get_states()
        state_names = [s["name"] for s in states]
        assert "trainingRound" in state_names


class TestAsyncWorkflowCreation:
    """
    Test AsyncDFL creation and initialization.

    Note: These tests patch CustomModelFactory.create_model since it requires
    a real TensorFlow model for async FL wrapping.
    """

    def test_init_workflow_returns_workflow(self, mock_node):
        """Test that __init__ returns a properly initialized workflow."""
        with patch("p2pfl.workflow.async_dfl.workflow.CustomModelFactory") as mock_factory:
            mock_factory.create_model.return_value = MagicMock()
            workflow = AsyncDFL(mock_node)

            assert isinstance(workflow, AsyncDFL)
            assert workflow.node == mock_node
            assert isinstance(workflow.peers, dict)
            assert workflow._machine is not None

    def test_init_workflow_initial_state(self, mock_node):
        """Test that initialized workflow starts in waitingSetup state."""
        with patch("p2pfl.workflow.async_dfl.workflow.CustomModelFactory") as mock_factory:
            mock_factory.create_model.return_value = MagicMock()
            workflow = AsyncDFL(mock_node)
            assert workflow.state == "waitingSetup"

    def test_init_workflow_does_not_register_commands(self, mock_node):
        """Test that __init__ does not register commands (start() does)."""
        with patch("p2pfl.workflow.async_dfl.workflow.CustomModelFactory") as mock_factory:
            mock_factory.create_model.return_value = MagicMock()
            AsyncDFL(mock_node)

            # Commands should NOT be registered at __init__ time; they are
            # registered when start() is called.
            mock_node.communication_protocol.add_command.assert_not_called()

    def test_init_workflow_wraps_model(self, mock_node):
        """Test that __init__ wraps the model for async FL."""
        with patch("p2pfl.workflow.async_dfl.workflow.CustomModelFactory") as mock_factory:
            mock_factory.create_model.return_value = MagicMock()
            AsyncDFL(mock_node)

            # Verify CustomModelFactory.create_model was called
            mock_factory.create_model.assert_called_once()

            # Verify set_model was called (to wrap with the result)
            mock_node.learner.set_model.assert_called_once()

    def test_workflow_node_address(self, mock_node):
        """Test that workflow has correct node address."""
        with patch("p2pfl.workflow.async_dfl.workflow.CustomModelFactory") as mock_factory:
            mock_factory.create_model.return_value = MagicMock()
            workflow = AsyncDFL(mock_node)
            assert workflow.node.address == mock_node.address


class TestAsyncWorkflowConditionsNoneGuards:
    """Tests that async workflow conditions handle None values safely."""

    def test_check_total_rounds_reached_returns_false_when_no_experiment(self, mock_node):
        """Test that check_total_rounds_reached returns False when no experiment is set."""
        with patch("p2pfl.workflow.async_dfl.workflow.CustomModelFactory") as mock_factory:
            mock_factory.create_model.return_value = MagicMock()
            workflow = AsyncDFL(mock_node)
            assert workflow.experiment is None
            assert workflow.check_total_rounds_reached() is False

    def test_check_iteration_network_updating_returns_true_at_round_zero(self, mock_node):
        """Test that check_iteration_network_updating returns True at round 0 (0 % tau == 0)."""
        with patch("p2pfl.workflow.async_dfl.workflow.CustomModelFactory") as mock_factory:
            mock_factory.create_model.return_value = MagicMock()
            workflow = AsyncDFL(mock_node)
            assert workflow.round == 0
            assert workflow.check_iteration_network_updating() is True

    def test_check_total_rounds_reached_returns_true_when_reached(self, mock_node):
        """Test that check_total_rounds_reached returns True when round >= total_rounds."""
        with patch("p2pfl.workflow.async_dfl.workflow.CustomModelFactory") as mock_factory:
            mock_factory.create_model.return_value = MagicMock()
            workflow = AsyncDFL(mock_node)
            with patch("p2pfl.workflow.engine.workflow.logger"):
                workflow.experiment = Experiment("test", 2, epochs_per_round=1)
                workflow.increase_round()
                workflow.increase_round()
                assert workflow.check_total_rounds_reached() is True


class TestAsyncWorkflowPeerState:
    """Tests for async peer state operations."""

    def test_peers_add_and_list(self):
        """Test adding peers to workflow."""
        peers: dict[str, AsyncPeerState] = {}
        peers["node_1"] = AsyncPeerState()
        peers["node_2"] = AsyncPeerState()
        assert "node_1" in peers
        assert "node_2" in peers

    def test_peers_clear(self):
        """Test clearing peers dict."""
        peers: dict[str, AsyncPeerState] = {}
        peers["node_1"] = AsyncPeerState()
        peers["node_2"] = AsyncPeerState()
        assert len(peers) == 2

        peers.clear()
        assert len(peers) == 0

    def test_add_loss(self):
        """Test add_loss extends and sets loss at round index."""
        peer = AsyncPeerState()
        peer.add_loss(0, 0.5)
        assert peer.losses == [0.5]
        peer.add_loss(3, 0.2)
        assert peer.losses == [0.5, 0.0, 0.0, 0.2]

    def test_default_values(self):
        """Test that AsyncPeerState has correct defaults."""
        peer = AsyncPeerState()
        assert peer.round_number == 0
        assert peer.push_sum_weight == 1.0
        assert peer.model is None
        assert peer.losses == []
        assert peer.push_time == 0
        assert peer.mixing_weight == 1.0
        assert peer.p2p_updating_idx == 0


class TestAsyncWorkflowLocalState:
    """Tests for async workflow experiment/round operations."""

    def test_workflow_experiment(self, mock_node):
        """Test workflow experiment management."""
        with patch("p2pfl.workflow.async_dfl.workflow.CustomModelFactory") as mock_factory:
            mock_factory.create_model.return_value = MagicMock()
            workflow = AsyncDFL(mock_node)
            workflow.experiment = Experiment("test_exp", 5, epochs_per_round=2)

            assert workflow.experiment.exp_name == "test_exp"
            assert workflow.experiment.total_rounds == 5
            assert workflow.experiment.epochs_per_round == 2
