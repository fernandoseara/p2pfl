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
"""Node tests."""

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from p2pfl.communication.protocols.protobuff.memory import MemoryCommunicationProtocol
from p2pfl.exceptions import LearnerRunningException, NodeRunningException
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.dataset.partition_strategies import RandomIIDPartitionStrategy
from p2pfl.learning.frameworks import Framework
from p2pfl.management.logger import logger
from p2pfl.node import Node
from p2pfl.node_state import NodeState, NodeStatus
from p2pfl.settings import Settings
from p2pfl.utils.utils import (
    check_equal_models,
    wait_convergence,
    wait_to_finish,
)
from p2pfl.workflow.engine.experiment import Experiment
from p2pfl.workflow.engine.workflow import WorkflowStatus

try:
    from p2pfl.examples.mnist.model.mlp_tensorflow import model_build_fn as model_build_fn_tensorflow
except ImportError:
    model_build_fn_tensorflow = pytest.param(None, marks=pytest.mark.skip(reason="TensorFlow not installed"))  # type: ignore[assignment]

try:
    from p2pfl.examples.mnist.model.mlp_pytorch import model_build_fn as model_build_fn_pytorch
except ImportError:
    model_build_fn_pytorch = pytest.param(None, marks=pytest.mark.skip(reason="PyTorch not installed"))  # type: ignore[assignment]


###
# Helpers
###


def _make_node():
    """Create a Node with a real PyTorch model but memory protocol (fast)."""
    from p2pfl.examples.mnist.model.mlp_pytorch import model_build_fn

    data = P2PFLDataset.from_huggingface("p2pfl/MNIST")
    partitions = data.generate_partitions(100, RandomIIDPartitionStrategy)
    return Node(model_build_fn(), partitions[0], protocol=MemoryCommunicationProtocol())


###
# Unit Tests: Lifecycle
###


class TestNodeLifecycle:
    """Node lifecycle tests."""

    @pytest.mark.asyncio
    async def test_start_twice_raises(self):
        """Starting a node that is already running raises NodeRunningException."""
        node = _make_node()
        await node.start()
        with pytest.raises(NodeRunningException):
            await node.start()
        await node.stop()

    @pytest.mark.asyncio
    async def test_stop_when_not_running_raises(self):
        """Stopping a node that is not running raises NodeRunningException."""
        node = _make_node()
        with pytest.raises(NodeRunningException):
            await node.stop()

    @pytest.mark.asyncio
    async def test_state_offline_before_start(self):
        """Node state is OFFLINE before start."""
        node = _make_node()
        assert node.state == NodeState.OFFLINE

    @pytest.mark.asyncio
    async def test_state_idle_after_start(self):
        """Node state is IDLE after start."""
        node = _make_node()
        await node.start()
        assert node.state == NodeState.IDLE
        await node.stop()

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Nodes can disconnect after connecting."""
        n1 = _make_node()
        n2 = _make_node()
        await n1.start()
        await n2.start()
        await n1.connect(n2.address)
        assert len(n1.get_neighbors(only_direct=True)) > 0
        await n1.disconnect(n2.address)
        await n1.stop()
        await n2.stop()


###
# Unit Tests: Properties
###


class TestNodeProperties:
    """Node property tests."""

    @pytest.mark.asyncio
    async def test_model_setter_blocked_during_learning(self):
        """Setting model during learning raises LearnerRunningException."""
        node = _make_node()
        await node.start()
        mock_wf = MagicMock()
        mock_wf.status.is_terminal = False
        mock_wf.status = WorkflowStatus.RUNNING
        node.workflow = mock_wf
        with pytest.raises(LearnerRunningException):
            node.model = MagicMock()
        node.workflow = None
        await node.stop()

    @pytest.mark.asyncio
    async def test_data_setter_blocked_during_learning(self):
        """Setting data during learning raises LearnerRunningException."""
        node = _make_node()
        await node.start()
        mock_wf = MagicMock()
        mock_wf.status = WorkflowStatus.RUNNING
        node.workflow = mock_wf
        with pytest.raises(LearnerRunningException):
            node.data = MagicMock()
        node.workflow = None
        await node.stop()

    @pytest.mark.asyncio
    async def test_status_snapshot(self):
        """Status snapshot returns correct values for idle node."""
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
        """Status reflects workflow state when a workflow is active."""
        node = _make_node()
        await node.start()
        mock_wf = MagicMock()
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


###
# Unit Tests: Learning
###


class TestNodeLearning:
    """Node learning tests."""

    @pytest.mark.asyncio
    async def test_set_stop_learning_when_no_workflow(self):
        """Stopping learning when no workflow is active does not raise."""
        node = _make_node()
        await node.start()
        await node.set_stop_learning()
        await node.stop()

    @pytest.mark.asyncio
    async def test_start_learning_while_running_raises(self):
        """Starting learning while already running raises NodeRunningException."""
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
        """Starting learning cleans up a previous terminal workflow."""
        node = _make_node()
        await node.start()
        mock_wf = MagicMock()
        mock_wf.status = MagicMock()
        mock_wf.status.is_terminal = True
        mock_wf.get_messages.return_value = {"msg1": MagicMock()}
        node.workflow = mock_wf
        exp = Experiment.create(exp_name="test", total_rounds=1, epochs_per_round=1)
        node.communication_protocol.remove_command = MagicMock()
        with contextlib.suppress(Exception):
            await node._start_learning_workflow("basic", exp)
        node.communication_protocol.remove_command.assert_called()
        node.workflow = None
        await node.stop()

    @pytest.mark.asyncio
    async def test_stop_during_learning_broadcasts_stop(self):
        """Stopping a node during learning transitions to OFFLINE."""
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
        """Stopping a node handles workflow stop errors gracefully."""
        node = _make_node()
        await node.start()
        mock_wf = MagicMock()
        mock_wf.stop = AsyncMock(side_effect=RuntimeError("workflow stop failed"))
        mock_wf.status = WorkflowStatus.RUNNING
        mock_wf.get_messages.return_value = {}
        node.workflow = mock_wf
        await node.stop()
        assert node.state == NodeState.OFFLINE


###
# Unit Tests: NodeState
###


class TestNodeState:
    """NodeState and NodeStatus tests."""

    def test_node_state_properties(self):
        """NodeState enum properties return correct values."""
        assert NodeState.OFFLINE.is_running is False
        assert NodeState.IDLE.is_running is True
        assert NodeState.LEARNING.is_learning is True
        assert NodeState.IDLE.is_learning is False
        assert NodeState.FINISHED.is_terminal is True
        assert NodeState.FAILED.is_terminal is True
        assert NodeState.CANCELLED.is_terminal is True
        assert NodeState.LEARNING.is_terminal is False

    def test_from_workflow_status(self):
        """NodeState.from_workflow_status maps correctly."""
        assert NodeState.from_workflow_status(WorkflowStatus.RUNNING) == NodeState.LEARNING
        assert NodeState.from_workflow_status(WorkflowStatus.FINISHED) == NodeState.FINISHED
        assert NodeState.from_workflow_status(WorkflowStatus.FAILED) == NodeState.FAILED
        assert NodeState.from_workflow_status(WorkflowStatus.CANCELLED) == NodeState.CANCELLED
        assert NodeState.from_workflow_status(WorkflowStatus.IDLE) == NodeState.IDLE

    def test_node_status_str(self):
        """NodeStatus string includes all relevant fields."""
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
        """NodeStatus string omits None fields."""
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


###
# E2E Tests: Convergence
###


@pytest.mark.e2e_train
@pytest.mark.asyncio
@pytest.mark.parametrize("x", [(2, 2), (6, 3)])
@pytest.mark.parametrize("model_build_fn", [model_build_fn_pytorch, model_build_fn_tensorflow])
async def test_convergence(x, model_build_fn):
    """Test convergence (on learning) of two nodes."""
    n, rounds = x

    Settings.general.SEED = 777
    Settings.heartbeat.TIMEOUT = 20

    # Data
    data = P2PFLDataset.from_huggingface("p2pfl/MNIST")
    partitions = data.generate_partitions(n * 50, RandomIIDPartitionStrategy)

    # Node Creation
    nodes = []
    for i in range(n):
        node = Node(model_build_fn(), partitions[i], protocol=MemoryCommunicationProtocol())
        await node.start()
        nodes.append(node)

    # Node Connection
    for i in range(len(nodes) - 1):
        await nodes[i + 1].connect(nodes[i].address)
        await asyncio.sleep(0.1)
    await wait_convergence(nodes, n - 1, only_direct=False)

    try:
        # Start Learning
        exp_name = await nodes[0].set_start_learning(rounds=rounds, epochs=1)

        # Wait
        await wait_to_finish(nodes, timeout=240)

        # Check if execution is correct
        for node in nodes:
            assert node.state == NodeState.FINISHED
            assert node.workflow.status == WorkflowStatus.FINISHED
            visited = [t["stage"] for t in node.workflow.stage_timings]
            assert visited[0] == "setup"
            assert "finish" in visited

        check_equal_models(nodes)

        # Get accuracies
        framework = nodes[0].model.get_framework()
        if framework == Framework.PYTORCH.value:
            accuracy_name = "test_metric"
        elif framework == Framework.TENSORFLOW.value:
            accuracy_name = "compile_metrics"
        else:
            raise ValueError(f"Framwork {framework} not known")

        # Select logs for this experiment
        global_logs = logger.get_global_logs()
        exp_logs = global_logs.get(exp_name)
        if exp_logs is None:
            raise ValueError(f"Experiment logs not found for exp={exp_name}")

        # collect per-node accuracy time series
        accuracies = [
            node_metrics[accuracy_name]
            for node_metrics in exp_logs.values()
            if accuracy_name in node_metrics and isinstance(node_metrics[accuracy_name], list | tuple)
        ]
        if not accuracies:
            pytest.fail(f"No '{accuracy_name}' metrics found in experiment logs for exp={exp_name}")

        # determine last round index dynamically (max round index across nodes)
        all_entries = [(idx, acc) for node_acc in accuracies for idx, acc in node_acc]
        if not all_entries:
            pytest.fail("No accuracy entries found in any node's metrics")

        last_round_idx = max(idx for idx, _ in all_entries)
        last_round_accuracies = [acc for idx, acc in all_entries if idx == last_round_idx]

        assert last_round_accuracies, "No accuracy values found for the last round"
        assert all(acc > 0.5 for acc in last_round_accuracies), f"Expected all accuracies > 0.5, got {last_round_accuracies}"

    finally:
        for node in nodes:
            await node.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("build_model_fn", [model_build_fn_pytorch, model_build_fn_tensorflow])
async def test_framework_node(build_model_fn):
    """Test a TensorFlow node."""
    # Data
    data = P2PFLDataset.from_huggingface("p2pfl/MNIST")
    partitions = data.generate_partitions(400, RandomIIDPartitionStrategy)

    # Create the model
    p2pfl_model = build_model_fn()

    # Nodes
    n1 = Node(p2pfl_model, partitions[0])
    n2 = Node(p2pfl_model.build_copy(), partitions[1])

    # Start
    await n1.start()
    await n2.start()

    # Connect
    await n2.connect(n1.address)
    await wait_convergence([n1, n2], 1, only_direct=True)

    # Start Learning
    await n1.set_start_learning(rounds=1, epochs=1)

    # Wait
    await wait_to_finish([n1, n2], timeout=120)

    # Check if execution is correct
    for node in [n1, n2]:
        assert node.state == NodeState.FINISHED
        assert node.state != NodeState.FAILED

    check_equal_models([n1, n2])

    # Stop
    await n1.stop()
    await n2.stop()
