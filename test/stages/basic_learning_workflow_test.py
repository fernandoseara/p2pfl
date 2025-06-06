from unittest.mock import MagicMock

import pytest

from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.stages.workflows.event_handler_workflow import BasicEventHandlerWorkflow
from p2pfl.stages.workflows.models import BasicLearningWorkflowModel
from p2pfl.stages.workflows.training_workflow import BasicTrainingWorkflow
from p2pfl.stages.workflows.workflows import LearningWorkflow


@pytest.fixture
def mock_node():
    node = MagicMock()

    # Mock LocalState
    local_state = MagicMock()
    local_state.get_experiment.return_value = MagicMock(exp_name="mock_exp")
    local_state.round = 0
    local_state.total_rounds = 3
    local_state.train_set = ["node_1", "node_2"]
    local_state.get_peer.return_value = "node_1"
    local_state.get_train_set.return_value = ["node_1", "node_2"]
    node.get_local_state.return_value = local_state

    # Mock Learner and Model
    model = MagicMock()
    model.get_weights.return_value = b"weights"
    model.round = 0
    learner = MagicMock()
    learner.get_model.return_value = model
    node.get_learner.return_value = learner

    # Mock CommunicationProtocol
    protocol = MagicMock()
    protocol.get_neighbors.return_value = ["node_2", "node_3"]
    protocol.set_neighbors = MagicMock()
    node.communication_protocol = protocol

    return node


@pytest.fixture
def workflow(mock_node):

    model = BasicLearningWorkflowModel(node=mock_node)
    machine = LearningWorkflow(model, BasicTrainingWorkflow(), BasicEventHandlerWorkflow())
    return model


@pytest.mark.asyncio
async def test_is_all_nodes_started_true(workflow):
    workflow.network_state.add_peer("node_1")
    workflow.network_state.add_peer("node_2")
    workflow.network_state.add_peer("node_3")
    assert workflow.is_all_nodes_started()


@pytest.mark.asyncio
async def test_get_partial_gossipping_candidates(workflow, mock_node):
    # Setup state
    workflow.network_state.add_peer("node_2")
    workflow.network_state.add_peer("node_3")
    workflow.node.get_local_state.return_value.train_set = ["node_1", "node_2", "node_3"]

    workflow.network_state.add_aggregated_from("node_2", "node_1")

    candidates = workflow.get_partial_gossipping_candidates()
    assert "node_3" in candidates
    assert "node_2" not in candidates


@pytest.mark.asyncio
async def test_is_all_votes_received(workflow):
    workflow.network_state.add_peer("node_1")
    workflow.network_state.add_peer("node_2")
    workflow.network_state.add_peer("node_3")

    workflow.node.communication_protocol.get_neighbors.return_value = ["node_2", "node_3"]

    for peer in ["node_1", "node_2", "node_3"]:
        workflow.network_state.add_vote(peer, "node_1", 1)

    assert workflow.is_all_votes_received()


@pytest.mark.asyncio
async def test_save_votes(workflow):
    await workflow.save_votes("node_2", 0, [("node_1", 1), ("node_3", 2)])
    votes = workflow.network_state.get_all_votes()["node_2"]
    assert votes["node_1"] == 1
    assert votes["node_3"] == 2


@pytest.mark.asyncio
async def test_save_full_model(workflow):
    weights = b"mock_weights"
    await workflow.save_full_model("node_2", 0, weights)
    assert workflow.node.get_learner().get_model.called
