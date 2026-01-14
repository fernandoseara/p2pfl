"""
Tests for basic learning workflow.

NOTE: This is a PoC / first draft - tests need to be expanded for full coverage.
The workflow architecture was refactored during the asyncio_support merge.
"""

from unittest.mock import MagicMock

import pytest

from p2pfl.stages.local_state.dfl_node_state import DFLLocalNodeState
from p2pfl.stages.network_state.basic_network_state import BasicNetworkState
from p2pfl.stages.workflows.models import BasicLearningWorkflowModel


@pytest.fixture
def mock_node():
    """Create a mock node for testing."""
    node = MagicMock()
    node.address = "node_1"

    # Mock LocalState
    local_state = MagicMock()
    local_state.get_experiment.return_value = MagicMock(exp_name="mock_exp")
    local_state.round = 0
    local_state.total_rounds = 3
    local_state.train_set = ["node_1", "node_2"]
    node.get_local_state.return_value = local_state

    # Mock Learner and Model
    model = MagicMock()
    model.get_parameters.return_value = [1.0, 2.0, 3.0]
    model.encode_parameters.return_value = b"weights"
    learner = MagicMock()
    learner.get_model.return_value = model
    learner.get_epochs.return_value = 1
    node.get_learner.return_value = learner

    # Mock CommunicationProtocol
    protocol = MagicMock()
    protocol.get_neighbors.return_value = {"node_2": {}, "node_3": {}}
    node.get_communication_protocol.return_value = protocol
    node.communication_protocol = protocol

    # Mock NodeWorkflow
    node_workflow = MagicMock()
    node_workflow.get_workflow_type.return_value = MagicMock(value="basic")
    node.get_node_workflow.return_value = node_workflow

    return node


@pytest.fixture
def local_state():
    """Create a local state for testing."""
    state = DFLLocalNodeState("node_1")
    return state


@pytest.fixture
def network_state():
    """Create a network state for testing."""
    return BasicNetworkState()


@pytest.fixture
def workflow(mock_node, local_state, network_state):
    """Create a workflow model for testing."""
    model = BasicLearningWorkflowModel(
        node=mock_node,
        local_state=local_state,
        network_state=network_state,
    )
    return model


# PoC Tests - basic sanity checks for the new architecture


@pytest.mark.asyncio
async def test_workflow_creation(workflow):
    """Test that workflow can be created."""
    assert workflow is not None
    assert workflow.network_state is not None
    assert workflow.local_state is not None


@pytest.mark.asyncio
async def test_network_state_add_peer(network_state):
    """Test adding peers to network state."""
    network_state.add_peer("node_1")
    network_state.add_peer("node_2")
    peers = network_state.list_peers()
    assert "node_1" in peers
    assert "node_2" in peers


@pytest.mark.asyncio
async def test_network_state_votes(network_state):
    """Test vote management in network state."""
    network_state.add_peer("node_1")
    network_state.add_peer("node_2")

    network_state.add_vote("node_1", "node_2", 1)
    network_state.add_vote("node_2", "node_1", 2)

    votes = network_state.get_all_votes()
    assert "node_1" in votes
    assert "node_2" in votes


@pytest.mark.asyncio
async def test_is_all_nodes_started(workflow):
    """Test is_all_nodes_started condition."""
    # Add peers to network state
    workflow.network_state.add_peer("node_1")
    workflow.network_state.add_peer("node_2")
    workflow.network_state.add_peer("node_3")

    # Mock get_neighbors to return 2 neighbors (+ self = 3 total)
    workflow.node.communication_protocol.get_neighbors.return_value = {"node_2": {}, "node_3": {}}

    assert workflow.is_all_nodes_started()


@pytest.mark.asyncio
async def test_local_state_experiment(local_state):
    """Test local state experiment management."""
    local_state.set_experiment("test_exp", 5, epochs_per_round=2)

    assert local_state.exp_name == "test_exp"
    assert local_state.total_rounds == 5
    assert local_state.epochs_per_round == 2
    assert local_state.epochs == 2  # alias property
