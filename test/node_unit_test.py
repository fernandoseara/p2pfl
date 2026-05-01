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
"""Unit tests for Node — fast, no real training, mock-heavy."""

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from p2pfl.communication.protocols.protobuff.memory import MemoryCommunicationProtocol
from p2pfl.exceptions import LearnerRunningException, NodeRunningException
from p2pfl.node import Node
from p2pfl.node_state import NodeState


def _make_node():
    """Create a Node with a real PyTorch model but memory protocol (fast)."""
    from p2pfl.examples.mnist.model.mlp_pytorch import model_build_fn
    from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
    from p2pfl.learning.dataset.partition_strategies import RandomIIDPartitionStrategy

    data = P2PFLDataset.from_huggingface("p2pfl/MNIST")
    partitions = data.generate_partitions(100, RandomIIDPartitionStrategy)
    return Node(model_build_fn(), partitions[0], protocol=MemoryCommunicationProtocol())


class TestNodeLifecycle:
    """Node Lifecycle tests."""

    @pytest.mark.asyncio
    async def test_start_twice_raises(self):
        """Test start twice raises."""
        node = _make_node()
        await node.start()
        with pytest.raises(NodeRunningException):
            await node.start()
        await node.stop()

    @pytest.mark.asyncio
    async def test_stop_when_not_running_raises(self):
        """Test stop when not running raises."""
        node = _make_node()
        with pytest.raises(NodeRunningException):
            await node.stop()

    @pytest.mark.asyncio
    async def test_state_offline_before_start(self):
        """Test state offline before start."""
        node = _make_node()
        assert node.state == NodeState.OFFLINE

    @pytest.mark.asyncio
    async def test_state_idle_after_start(self):
        """Test state idle after start."""
        node = _make_node()
        await node.start()
        assert node.state == NodeState.IDLE
        await node.stop()

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test disconnect."""
        n1 = _make_node()
        n2 = _make_node()
        await n1.start()
        await n2.start()
        await n1.connect(n2.address)
        assert len(n1.get_neighbors(only_direct=True)) > 0
        await n1.disconnect(n2.address)
        await n1.stop()
        await n2.stop()


class TestNodeProperties:
    """Node Properties tests."""

    @pytest.mark.asyncio
    async def test_model_setter_blocked_during_learning(self):
        """Test model setter blocked during learning."""
        node = _make_node()
        await node.start()
        # Simulate learning state
        mock_wf = MagicMock()
        mock_wf.status.is_terminal = False
        from p2pfl.workflow.engine.workflow import WorkflowStatus

        mock_wf.status = WorkflowStatus.RUNNING
        node.workflow = mock_wf
        with pytest.raises(LearnerRunningException):
            node.model = MagicMock()
        node.workflow = None
        await node.stop()

    @pytest.mark.asyncio
    async def test_data_setter_blocked_during_learning(self):
        """Test data setter blocked during learning."""
        node = _make_node()
        await node.start()
        mock_wf = MagicMock()
        from p2pfl.workflow.engine.workflow import WorkflowStatus

        mock_wf.status = WorkflowStatus.RUNNING
        node.workflow = mock_wf
        with pytest.raises(LearnerRunningException):
            node.data = MagicMock()
        node.workflow = None
        await node.stop()

    @pytest.mark.asyncio
    async def test_status_snapshot(self):
        """Test status snapshot."""
        node = _make_node()
        await node.start()
        status = node.status
        assert status.address == node.address
        assert status.state == NodeState.IDLE
        assert status.experiment is None
        assert status.error is None
        assert status.current_stage_name is None
        await node.stop()

    @pytest.mark.asyncio
    async def test_status_with_workflow(self):
        """Test status with workflow."""
        node = _make_node()
        await node.start()
        mock_wf = MagicMock()
        from p2pfl.workflow.engine.workflow import WorkflowStatus

        mock_wf.status = WorkflowStatus.FAILED
        mock_wf.error = RuntimeError("boom")
        mock_wf.current_stage_name = "voting"
        mock_wf.experiment = MagicMock()
        node.workflow = mock_wf
        status = node.status
        assert status.state == NodeState.FAILED
        assert status.error == "boom"
        assert status.current_stage_name == "voting"
        node.workflow = None
        await node.stop()


class TestNodeLearning:
    """Node Learning tests."""

    @pytest.mark.asyncio
    async def test_set_stop_learning_when_no_workflow(self):
        """Test set stop learning when no workflow."""
        node = _make_node()
        await node.start()
        await node.set_stop_learning()
        await node.stop()

    @pytest.mark.asyncio
    async def test_start_learning_while_running_raises(self):
        """Test start learning while running raises."""
        node = _make_node()
        await node.start()
        mock_wf = MagicMock()
        mock_wf.status = MagicMock()
        mock_wf.status.is_terminal = False
        node.workflow = mock_wf
        with pytest.raises(NodeRunningException):
            await node._start_learning_workflow("basic", MagicMock())
        node.workflow = None
        await node.stop()

    @pytest.mark.asyncio
    async def test_start_learning_cleans_terminal_workflow(self):
        """Test start learning cleans terminal workflow."""
        node = _make_node()
        await node.start()
        # Simulate a terminal workflow that should be cleaned up
        mock_wf = MagicMock()
        mock_wf.status = MagicMock()
        mock_wf.status.is_terminal = True
        mock_wf.get_messages.return_value = {"msg1": MagicMock()}
        node.workflow = mock_wf
        # Calling _start_learning_workflow should clean up the old one
        from p2pfl.workflow.engine.experiment import Experiment

        exp = Experiment.create(exp_name="test", total_rounds=1, epochs_per_round=1)
        # It will create a new workflow, but fail because no neighbors — that's fine,
        # we just want to verify the old workflow was cleaned up
        node.communication_protocol.remove_command = MagicMock()
        with contextlib.suppress(Exception):
            await node._start_learning_workflow("basic", exp)
        # The old workflow commands should have been unregistered
        node.communication_protocol.remove_command.assert_called()
        node.workflow = None
        await node.stop()

    @pytest.mark.asyncio
    async def test_stop_during_learning_broadcasts_stop(self):
        """Test stop during learning broadcasts stop."""
        node = _make_node()
        await node.start()
        n2 = _make_node()
        await n2.start()
        await node.connect(n2.address)
        await asyncio.sleep(0.1)

        await node.set_start_learning(rounds=100, epochs=1)
        await asyncio.sleep(0.5)
        assert node.state == NodeState.LEARNING
        await node.stop()
        assert node.state == NodeState.OFFLINE

        await n2.stop()

    @pytest.mark.asyncio
    async def test_stop_workflow_error_handled(self):
        """Test stop workflow error handled."""
        node = _make_node()
        await node.start()
        mock_wf = MagicMock()
        mock_wf.stop = AsyncMock(side_effect=RuntimeError("workflow stop failed"))
        from p2pfl.workflow.engine.workflow import WorkflowStatus

        mock_wf.status = WorkflowStatus.RUNNING
        mock_wf.get_messages.return_value = {}
        node.workflow = mock_wf
        # stop should not raise despite workflow error
        await node.stop()
        assert node.state == NodeState.OFFLINE


class TestNodeState:
    """Node State tests."""

    def test_node_state_properties(self):
        """Test node state properties."""
        assert NodeState.OFFLINE.is_running is False
        assert NodeState.IDLE.is_running is True
        assert NodeState.LEARNING.is_learning is True
        assert NodeState.IDLE.is_learning is False
        assert NodeState.FINISHED.is_terminal is True
        assert NodeState.FAILED.is_terminal is True
        assert NodeState.CANCELLED.is_terminal is True
        assert NodeState.LEARNING.is_terminal is False

    def test_from_workflow_status(self):
        """Test from workflow status."""
        from p2pfl.workflow.engine.workflow import WorkflowStatus

        assert NodeState.from_workflow_status(WorkflowStatus.RUNNING) == NodeState.LEARNING
        assert NodeState.from_workflow_status(WorkflowStatus.FINISHED) == NodeState.FINISHED
        assert NodeState.from_workflow_status(WorkflowStatus.FAILED) == NodeState.FAILED
        assert NodeState.from_workflow_status(WorkflowStatus.CANCELLED) == NodeState.CANCELLED
        assert NodeState.from_workflow_status(WorkflowStatus.IDLE) == NodeState.IDLE

    def test_node_status_str(self):
        """Test node status str."""
        from p2pfl.node_state import NodeStatus

        s = NodeStatus(
            address="127.0.0.1:8000",
            state=NodeState.FAILED,
            num_neighbors=3,
            experiment=MagicMock(__str__=lambda self: "exp1"),
            error="timeout",
            current_stage_name="voting",
        )
        text = str(s)
        assert "127.0.0.1:8000" in text
        assert "failed" in text
        assert "timeout" in text
        assert "voting" in text
        assert "neighbors=3" in text

    def test_node_status_str_minimal(self):
        """Test node status str minimal."""
        from p2pfl.node_state import NodeStatus

        s = NodeStatus(
            address="node1",
            state=NodeState.IDLE,
            num_neighbors=0,
            experiment=None,
            error=None,
            current_stage_name=None,
        )
        text = str(s)
        assert "node1" in text
        assert "experiment" not in text
        assert "error" not in text
