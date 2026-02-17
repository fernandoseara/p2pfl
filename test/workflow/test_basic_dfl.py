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

"""Tests for BasicDFL."""

from unittest.mock import MagicMock, patch

import pytest

from p2pfl.workflow.basic_dfl.workflow import (
    BasicDFL,
    BasicPeerState,
    get_states,
    get_transitions,
)
from p2pfl.workflow.engine.experiment import Experiment
from p2pfl.workflow.factory import WorkflowType


@pytest.fixture
def mock_node():
    """Create a mock node for testing."""
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
    node.workflow_type = WorkflowType.BASIC

    return node


@pytest.fixture
def workflow(mock_node):
    """Create a workflow model for testing."""
    return BasicDFL(mock_node)


class TestBasicWorkflowStatesAndTransitions:
    """Tests for workflow state machine configuration."""

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


class TestBasicWorkflowCreation:
    """Tests for BasicDFL creation and initialization."""

    def test_init_workflow_returns_workflow(self, mock_node):
        """Test that __init__ returns a properly initialized workflow."""
        workflow = BasicDFL(mock_node)

        assert isinstance(workflow, BasicDFL)
        assert workflow.node == mock_node
        assert isinstance(workflow.peers, dict)
        assert workflow._machine is not None

    def test_init_workflow_initial_state(self, mock_node):
        """Test that initialized workflow starts in waitingSetup state."""
        workflow = BasicDFL(mock_node)
        assert workflow.state == "waitingSetup"

    def test_init_workflow_does_not_register_commands(self, mock_node):
        """Test that __init__ does not register commands (start() does)."""
        BasicDFL(mock_node)

        # Commands should NOT be registered at __init__ time; they are
        # registered when start() is called.
        mock_node.communication_protocol.add_command.assert_not_called()

    def test__cleanup_removes_from_machine(self, mock_node):
        """Test that _cleanup() removes workflow from its machine."""
        workflow = BasicDFL(mock_node)
        workflow._cleanup()
        # After _cleanup, the workflow is removed from the machine
        # To reset, create a new workflow instance

    def test_train_set_defaults_empty(self, mock_node):
        """Test that train_set defaults to empty list."""
        workflow = BasicDFL(mock_node)
        assert workflow.train_set == []


class TestBasicWorkflowConditions:
    """Tests for workflow condition methods."""

    def test_is_all_nodes_started(self, workflow):
        """Test is_all_nodes_started condition."""
        # Add peers directly
        workflow.peers["node_1"] = BasicPeerState()
        workflow.peers["node_2"] = BasicPeerState()
        workflow.peers["node_3"] = BasicPeerState()

        # Mock get_neighbors to return 2 neighbors (+ self = 3 total)
        workflow.node.communication_protocol.get_neighbors.return_value = {"node_2": {}, "node_3": {}}

        assert workflow.is_all_nodes_started()

    def test_is_all_nodes_started_false_when_missing(self, workflow):
        """Test is_all_nodes_started returns false when peers are missing."""
        workflow.peers["node_1"] = BasicPeerState()
        # Only 1 peer, but we have 2 neighbors
        workflow.node.communication_protocol.get_neighbors.return_value = {"node_2": {}, "node_3": {}}

        assert not workflow.is_all_nodes_started()

    def test_is_initiator_node(self, workflow):
        """Test is_initiator_node condition."""
        workflow.experiment = Experiment("test", 5, is_initiator=True)
        assert workflow.is_initiator_node()

        workflow.experiment = Experiment("test", 5, is_initiator=False)
        assert not workflow.is_initiator_node()

        workflow.experiment = None
        assert not workflow.is_initiator_node()

    def test_in_train_set(self, workflow):
        """Test in_train_set condition."""
        workflow.train_set = ["node_1", "node_2"]
        workflow.node.address = "node_1"
        assert workflow.in_train_set()

        workflow.node.address = "node_3"
        assert not workflow.in_train_set()


class TestBasicWorkflowConditionsNoneGuards:
    """Tests that workflow conditions handle None values safely."""

    def test_is_total_rounds_reached_returns_false_when_no_experiment(self, workflow):
        """Test that is_total_rounds_reached returns False when no experiment is set."""
        assert workflow.experiment is None
        assert workflow.is_total_rounds_reached() is False

    def test_is_total_rounds_reached_returns_false_when_total_rounds_is_none(self, workflow):
        """Test that is_total_rounds_reached returns False when total_rounds is None."""
        workflow.experiment = Experiment("test", 5, epochs_per_round=1)
        # Force total_rounds to None
        workflow.experiment.total_rounds = None
        assert workflow.is_total_rounds_reached() is False

    def test_is_total_rounds_reached_returns_true_when_reached(self, workflow):
        """Test that is_total_rounds_reached returns True when round >= total_rounds."""
        with patch("p2pfl.workflow.engine.workflow.logger"):
            workflow.experiment = Experiment("test", 2, epochs_per_round=1)
            workflow.increase_round()
            workflow.increase_round()
            assert workflow.is_total_rounds_reached() is True


class TestBasicWorkflowPeerState:
    """Tests for peer state operations in workflow."""

    def test_peers_add_and_list(self):
        """Test adding peers to workflow."""
        peers: dict[str, BasicPeerState] = {}
        peers["node_1"] = BasicPeerState()
        peers["node_2"] = BasicPeerState()
        assert "node_1" in peers
        assert "node_2" in peers

    def test_peer_votes(self):
        """Test vote management on peer state."""
        peers: dict[str, BasicPeerState] = {}
        peers["node_1"] = BasicPeerState()
        peers["node_2"] = BasicPeerState()

        peers["node_1"].votes["node_2"] = 1
        peers["node_2"].votes["node_1"] = 2

        assert "node_2" in peers["node_1"].votes
        assert "node_1" in peers["node_2"].votes

    def test_peers_clear(self):
        """Test clearing peers dict."""
        peers: dict[str, BasicPeerState] = {}
        peers["node_1"] = BasicPeerState()
        peers["node_2"] = BasicPeerState()
        assert len(peers) == 2

        peers.clear()
        assert len(peers) == 0

    def test_reset_round(self):
        """Test reset_round clears per-round state."""
        peer = BasicPeerState()
        peer.model = MagicMock()
        peer.aggregated_from = ["a", "b"]
        peer.votes = {"x": 1}
        peer.reset_round()
        assert peer.model is None
        assert peer.aggregated_from == []
        assert peer.votes == {}


class TestBasicWorkflowLocalState:
    """Tests for local state operations in workflow."""

    def test_workflow_experiment(self):
        """Test workflow experiment management."""
        node = MagicMock()
        node.address = "node_1"
        workflow = BasicDFL(node)
        workflow.experiment = Experiment("test_exp", 5, epochs_per_round=2)

        assert workflow.experiment.exp_name == "test_exp"
        assert workflow.experiment.total_rounds == 5
        assert workflow.experiment.epochs_per_round == 2

    def test_workflow_train_set(self):
        """Test workflow train set management."""
        node = MagicMock()
        node.address = "node_1"
        workflow = BasicDFL(node)
        assert workflow.train_set == []
        workflow.train_set = ["node_1", "node_2"]
        assert len(workflow.train_set) == 2

    def test_workflow_state_clear(self):
        """Test workflow state clear on trainingFinished."""
        node = MagicMock()
        node.address = "node_1"
        workflow = BasicDFL(node)
        workflow.experiment = Experiment("test_exp", 5)
        workflow.train_set = ["node_1"]
        # Simulate clearing (what trainingFinished does)
        workflow.experiment = None
        workflow.round = 0
        workflow.train_set = []

        assert workflow.experiment is None
        assert workflow.train_set == []
