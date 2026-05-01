#
# This file is part of the p2pfl (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2022 Pedro Guijas Bravo.
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
"""Utils tests."""

import random
from unittest.mock import AsyncMock, MagicMock, call

import numpy as np
import pytest

from p2pfl.learning.frameworks import Framework
from p2pfl.utils.node_component import NodeComponent, allow_no_addr_check
from p2pfl.utils.seed import set_seed
from p2pfl.utils.topologies import TopologyFactory, TopologyType

###
# Topology Tests
###


# Mock Node class for testing
class MockNode:
    """Mock Node class for testing."""

    def __init__(self, address):
        """Initialize the mock node."""
        self.address = address
        self.connect = MagicMock()


@pytest.mark.parametrize(
    "topology_type, expected_matrix",
    [
        (TopologyType.STAR, np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])),
        (TopologyType.FULL, np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])),
        (TopologyType.LINE, np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])),
        (TopologyType.RING, np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])),
    ],
)
def test_generate_matrix(topology_type, expected_matrix):
    """Test the generation of adjacency matrices for different topologies."""
    num_nodes = 4
    matrix = TopologyFactory.generate_matrix(topology_type, num_nodes)
    assert np.array_equal(matrix, expected_matrix)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "adjacency_matrix, expected_calls",
    [
        (
            np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]),  # Star topology matrix
            [
                [call("address_1"), call("address_2"), call("address_3")],
                [],
                [],
                [],
            ],
        ),
        (
            np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]),  # Full topology matrix
            [
                [call("address_1"), call("address_2"), call("address_3")],
                [call("address_2"), call("address_3")],
                [call("address_3")],
                [],
            ],
        ),
    ],
)
async def test_connect_nodes(adjacency_matrix, expected_calls):
    """Test that nodes are connected according to the provided adjacency matrix."""
    num_nodes = adjacency_matrix.shape[0]  # Get num_nodes from matrix shape
    nodes = [MockNode(f"address_{i}") for i in range(num_nodes)]
    # Replace connect with AsyncMock
    for node in nodes:
        node.connect = AsyncMock()

    await TopologyFactory.connect_nodes(adjacency_matrix, nodes)

    for i, calls in enumerate(expected_calls):
        nodes[i].connect.assert_has_calls(calls, any_order=True)
        assert nodes[i].connect.call_count == len(calls)


def test_invalid_topology_type():
    """Test that an exception is raised when an invalid topology type is passed."""
    with pytest.raises(ValueError):
        TopologyFactory.generate_matrix("invalid_type", 4)


@pytest.mark.parametrize(
    "topology_type, num_nodes",
    [
        (TopologyType.RANDOM_2, 0),
        (TopologyType.RANDOM_2, 1),
        (TopologyType.RANDOM_2, 2),
        (TopologyType.RANDOM_2, 10),
        (TopologyType.RANDOM_3, 0),
        (TopologyType.RANDOM_3, 1),
        (TopologyType.RANDOM_3, 2),
        (TopologyType.RANDOM_3, 10),
        (TopologyType.RANDOM_4, 0),
        (TopologyType.RANDOM_4, 1),
        (TopologyType.RANDOM_4, 2),
        (TopologyType.RANDOM_4, 10),
    ],
)
def test_generate_random_matrix_properties(topology_type, num_nodes):
    """Test properties of randomly generated adjacency matrices."""
    matrix = TopologyFactory.generate_matrix(topology_type, num_nodes)

    # Check shape
    assert matrix.shape == (num_nodes, num_nodes)

    # Check diagonal is zero
    assert np.all(np.diag(matrix) == 0)

    # Check symmetry
    assert np.array_equal(matrix, matrix.T)

    # Check number of edges
    if num_nodes <= 1:
        expected_num_edges = 0
    else:
        if topology_type == TopologyType.RANDOM_2:
            avg_degree = 2
        elif topology_type == TopologyType.RANDOM_3:
            avg_degree = 3
        else:  # RANDOM_4
            avg_degree = 4

        # Calculate expected number of edges based on implementation logic
        num_edges_target = round(num_nodes * avg_degree / 2)
        max_possible_edges = num_nodes * (num_nodes - 1) // 2
        expected_num_edges = min(num_edges_target, max_possible_edges)

    actual_num_edges = np.sum(matrix) // 2
    assert actual_num_edges == expected_num_edges


###
# NodeComponent Tests
###


class MockNodeComponent(NodeComponent):
    """Mock class inheriting from NodeComponent for testing."""

    def __init__(self):
        """Initialize the mock class."""
        # super init
        super().__init__()

    def example_method(self) -> str:
        """Return the address. Example method that requires addr to be set."""
        return self.address

    @allow_no_addr_check
    def get_default_name(self) -> str:
        """Return "Hola!". A method with no addr check."""
        return "Hola!"


def test_node_component_initialization():
    """Test initial state and setting of addr."""
    component = MockNodeComponent()
    assert component.address == ""

    address = "test_address"
    returned_addr = component.set_address(address)
    assert component.address == address
    assert returned_addr == address


def test_node_component_methods():
    """Test method calls with and without addr set."""
    component = MockNodeComponent()
    assert component.get_default_name() == "Hola!"

    # Method call without addr should raise ValueError
    with pytest.raises(ValueError):
        component.example_method()

    # Method call with addr set should succeed
    addr = "test_address"
    component.set_address(addr)
    assert component.example_method() == addr


###
# Seed Tests
###


def test_set_seed_none_is_noop():
    """set_seed(None) should return immediately without changing any state."""
    before_py = random.getstate()
    before_np = np.random.get_state()
    set_seed(None)
    assert random.getstate() == before_py
    # NumPy state is a tuple; compare the key[1] array
    assert np.array_equal(before_np[1], np.random.get_state()[1])


def test_set_seed_determinism():
    """set_seed with a value makes Python and NumPy RNGs deterministic."""
    set_seed(42, framework=Framework.PYTORCH)
    py_vals = [random.random() for _ in range(5)]
    np_vals = list(np.random.rand(5))

    set_seed(42, framework=Framework.PYTORCH)
    assert [random.random() for _ in range(5)] == py_vals
    assert list(np.random.rand(5)) == np_vals


def test_set_seed_pytorch():
    """set_seed seeds PyTorch when framework=PYTORCH."""
    torch = pytest.importorskip("torch")
    set_seed(123, framework=Framework.PYTORCH)
    t1 = torch.rand(3)
    set_seed(123, framework=Framework.PYTORCH)
    t2 = torch.rand(3)
    assert torch.equal(t1, t2)


def test_set_seed_pytorch_via_string():
    """set_seed accepts framework as a string."""
    torch = pytest.importorskip("torch")
    set_seed(99, framework="pytorch")
    t1 = torch.rand(3)
    set_seed(99, framework="pytorch")
    t2 = torch.rand(3)
    assert torch.equal(t1, t2)


def test_set_seed_tensorflow():
    """set_seed seeds TensorFlow when framework=TENSORFLOW."""
    tf = pytest.importorskip("tensorflow")
    set_seed(77, framework=Framework.TENSORFLOW)
    t1 = tf.random.stateless_normal([3], seed=[77, 0])
    set_seed(77, framework=Framework.TENSORFLOW)
    t2 = tf.random.stateless_normal([3], seed=[77, 0])
    assert tf.reduce_all(tf.equal(t1, t2)).numpy()


###
# Utils function tests (p2pfl.utils.utils)
###


class TestWaitConvergence:
    """Tests for wait_convergence utility."""

    @pytest.mark.asyncio
    async def test_wait_convergence_immediate(self):
        """Nodes already have enough neighbors -- converges immediately."""
        from p2pfl.utils.utils import wait_convergence

        n1 = MagicMock()
        n1.get_neighbors.return_value = ["b", "c"]
        n1.address = "a"
        n2 = MagicMock()
        n2.get_neighbors.return_value = ["a", "c"]
        n2.address = "b"
        await wait_convergence([n1, n2], n_neis=2, wait=1)

    @pytest.mark.asyncio
    async def test_wait_convergence_timeout_raises(self):
        """wait_convergence raises AssertionError on timeout."""
        from p2pfl.utils.utils import wait_convergence

        node = MagicMock()
        node.get_neighbors.return_value = []
        node.address = "a"
        with pytest.raises(AssertionError):
            await wait_convergence([node], n_neis=5, wait=0.1)

    @pytest.mark.asyncio
    async def test_wait_convergence_with_debug(self):
        """wait_convergence with debug=True prints connectivity matrix."""
        from p2pfl.utils.utils import wait_convergence

        n1 = MagicMock()
        n1.get_neighbors.return_value = ["b"]
        n1.address = "a"
        n2 = MagicMock()
        n2.get_neighbors.return_value = ["a"]
        n2.address = "b"
        await wait_convergence([n1, n2], n_neis=1, wait=2, debug=True)


class TestPrintConnectivityMatrix:
    """Tests for _print_connectivity_matrix utility."""

    def test_print_connectivity_matrix_final(self):
        """Test _print_connectivity_matrix with final=True."""
        from p2pfl.utils.utils import _print_connectivity_matrix

        n1 = MagicMock()
        n1.get_neighbors.return_value = ["b"]
        n1.address = "a"
        n2 = MagicMock()
        n2.get_neighbors.return_value = ["a"]
        n2.address = "b"
        # Should not raise
        _print_connectivity_matrix([n1, n2], final=True)

    def test_print_connectivity_matrix_not_final(self):
        """Test _print_connectivity_matrix with final=False."""
        from p2pfl.utils.utils import _print_connectivity_matrix

        n1 = MagicMock()
        n1.get_neighbors.return_value = ["b"]
        n1.address = "a"
        n2 = MagicMock()
        n2.get_neighbors.return_value = ["a"]
        n2.address = "b"
        _print_connectivity_matrix([n1, n2], final=False)

    def test_print_connectivity_matrix_long_address(self):
        """Test truncation of long addresses."""
        from p2pfl.utils.utils import _print_connectivity_matrix

        n1 = MagicMock()
        n1.get_neighbors.return_value = []
        n1.address = "a" * 30  # Long address, triggers truncation
        _print_connectivity_matrix([n1], final=True)

    def test_print_connectivity_matrix_non_uniform(self):
        """Test non-uniform topology reporting."""
        from p2pfl.utils.utils import _print_connectivity_matrix

        n1 = MagicMock()
        n1.get_neighbors.return_value = ["b", "c"]
        n1.address = "a"
        n2 = MagicMock()
        n2.get_neighbors.return_value = ["a"]
        n2.address = "b"
        n3 = MagicMock()
        n3.get_neighbors.return_value = []
        n3.address = "c"
        _print_connectivity_matrix([n1, n2, n3], final=False)


class TestFullConnection:
    """Tests for full_connection utility."""

    @pytest.mark.asyncio
    async def test_full_connection(self):
        """full_connection connects node to all other nodes."""
        from p2pfl.utils.utils import full_connection

        main = MagicMock()
        main.connect = AsyncMock()
        main.address = "main"
        others = []
        for i in range(3):
            n = MagicMock()
            n.address = f"node_{i}"
            others.append(n)
        await full_connection(main, others)
        assert main.connect.call_count == 3


class TestNodeLearningError:
    """Tests for NodeLearningError."""

    def test_error_message(self):
        """Error message includes all failed node addresses."""
        from p2pfl.utils.utils import NodeLearningError

        err = NodeLearningError([("node1", ValueError("bad")), ("node2", RuntimeError("crash"))])
        assert "node1" in str(err)
        assert "node2" in str(err)
        assert len(err.failed_nodes) == 2


class _FakeNode:
    """Minimal fake node for wait_to_finish tests using real NodeState enums."""

    def __init__(self, address, state, workflow=None):
        self.address = address
        self.state = state
        self.workflow = workflow


class TestWaitToFinish:
    """Tests for wait_to_finish utility."""

    @pytest.mark.asyncio
    async def test_wait_to_finish_all_finished(self):
        """wait_to_finish returns when all nodes are in terminal state."""
        from p2pfl.node_state import NodeState
        from p2pfl.utils.utils import wait_to_finish

        n1 = _FakeNode("n1", NodeState.FINISHED)
        await wait_to_finish([n1], timeout=5)

    @pytest.mark.asyncio
    async def test_wait_to_finish_timeout(self):
        """wait_to_finish raises TimeoutError if nodes don't finish."""
        from p2pfl.node_state import NodeState
        from p2pfl.utils.utils import wait_to_finish

        wf = MagicMock()
        wf.status.value = "running"
        wf.current_stage_name = "train"
        n1 = _FakeNode("n1", NodeState.LEARNING, workflow=wf)
        with pytest.raises(TimeoutError):
            await wait_to_finish([n1], timeout=0.1, debug=True)

    @pytest.mark.asyncio
    async def test_wait_to_finish_failed_raises(self):
        """wait_to_finish raises NodeLearningError when a node failed."""
        from p2pfl.node_state import NodeState
        from p2pfl.utils.utils import NodeLearningError, wait_to_finish

        wf = MagicMock()
        wf.error = RuntimeError("training crash")
        n1 = _FakeNode("n1", NodeState.FAILED, workflow=wf)
        with pytest.raises(NodeLearningError):
            await wait_to_finish([n1], timeout=5, raise_on_error=True)

    @pytest.mark.asyncio
    async def test_wait_to_finish_failed_no_raise(self):
        """wait_to_finish with raise_on_error=False does not raise."""
        from p2pfl.node_state import NodeState
        from p2pfl.utils.utils import wait_to_finish

        wf = MagicMock()
        wf.error = RuntimeError("crash")
        n1 = _FakeNode("n1", NodeState.FAILED, workflow=wf)
        await wait_to_finish([n1], timeout=5, raise_on_error=False)


class TestCheckEqualModels:
    """Tests for check_equal_models utility."""

    def test_check_equal_models_same(self):
        """check_equal_models passes for identical model parameters."""
        from p2pfl.utils.utils import check_equal_models

        params = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        n1 = MagicMock()
        n1.model.get_parameters.return_value = [p.copy() for p in params]
        n2 = MagicMock()
        n2.model.get_parameters.return_value = [p.copy() for p in params]
        check_equal_models([n1, n2])

    def test_check_equal_models_none_params_raises(self):
        """check_equal_models raises ValueError when first node has None params."""
        from p2pfl.utils.utils import check_equal_models

        n1 = MagicMock()
        n1.model.get_parameters.return_value = None
        n2 = MagicMock()
        n2.model.get_parameters.return_value = [np.array([1.0])]
        with pytest.raises(ValueError, match="Model parameters are None"):
            check_equal_models([n1, n2])
