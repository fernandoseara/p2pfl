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

import asyncio  # noqa: E402, I001
import pytest  # noqa: E402
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset  # noqa: E402
from p2pfl.learning.dataset.partition_strategies import RandomIIDPartitionStrategy  # noqa: E402
from p2pfl.learning.frameworks import Framework
from p2pfl.management.logger import logger  # noqa: E402
from p2pfl.node import Node  # noqa: E402
from p2pfl.node_state import NodeState
from p2pfl.communication.protocols.protobuff.memory import MemoryCommunicationProtocol
from p2pfl.settings import Settings
from p2pfl.workflow.engine.workflow import WorkflowStatus
from p2pfl.utils.utils import (  # noqa: E402
    check_equal_models,
    wait_convergence,
    wait_to_finish,
)

try:
    from p2pfl.examples.mnist.model.mlp_tensorflow import model_build_fn as model_build_fn_tensorflow
except ImportError:
    model_build_fn_tensorflow = pytest.param(None, marks=pytest.mark.skip(reason="TensorFlow not installed"))  # type: ignore[assignment]

try:
    from p2pfl.examples.mnist.model.mlp_pytorch import model_build_fn as model_build_fn_pytorch
except ImportError:
    model_build_fn_pytorch = pytest.param(None, marks=pytest.mark.skip(reason="PyTorch not installed"))  # type: ignore[assignment]


###
# Tests Learning
###


# TODO: Add more frameworks and aggregators
#
#   Really important note: When training (pytorch) with a fixed seed and the process is shared, different training speeds affect to the
#   stochastic process, so is not fully deterministic!.
#
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
        # Flatten all (idx, acc) pairs and find max index, then collect corresponding accuracies
        all_entries = [(idx, acc) for node_acc in accuracies for idx, acc in node_acc]
        if not all_entries:
            pytest.fail("No accuracy entries found in any node's metrics")

        last_round_idx = max(idx for idx, _ in all_entries)
        last_round_accuracies = [acc for idx, acc in all_entries if idx == last_round_idx]

        assert last_round_accuracies, "No accuracy values found for the last round"
        assert all(acc > 0.5 for acc in last_round_accuracies), f"Expected all accuracies > 0.5, got {last_round_accuracies}"

    finally:
        # Stop Nodes
        for n in nodes:
            await n.stop()


###
# Training with other frameworks
###


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
