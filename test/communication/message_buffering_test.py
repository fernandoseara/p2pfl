#
# This file is part of the p2pfl (see https://github.com/pguijas/p2pfl).
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
"""
Tests for message buffering and deferred replay.

When workflow messages (e.g. hello) arrive at a node before start_learning
has propagated to it, the node hasn't registered workflow commands yet.
The server buffers these messages and replays them once the commands are
registered. The replay must happen after the workflow status is RUNNING
so the WorkflowCommand.execute() guard (is_learning) passes.
"""

import asyncio
import contextlib

import pytest

from p2pfl.communication.commands.command import Command
from p2pfl.communication.protocols.protobuff.memory import MemoryCommunicationProtocol
from p2pfl.communication.protocols.protobuff.proto import node_pb2
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.dataset.partition_strategies import RandomIIDPartitionStrategy
from p2pfl.node import Node
from p2pfl.settings import Settings
from p2pfl.utils.topologies import TopologyFactory, TopologyType
from p2pfl.utils.utils import wait_convergence, wait_to_finish
from p2pfl.workflow.engine.context import WorkflowContext
from p2pfl.workflow.engine.message import on_message
from p2pfl.workflow.engine.stage import Stage
from p2pfl.workflow.engine.workflow import Workflow
from p2pfl.workflow.factory import register_workflow


class TrackingCommand(Command):
    """Command that records every invocation."""

    def __init__(self, name: str = "buffered_cmd") -> None:
        """Initialize the tracking command."""
        self._name = name
        self.received: list[tuple[str, tuple]] = []

    def get_name(self) -> str:  # type: ignore[override]
        """Get the name of the command."""
        return self._name

    async def execute(self, source: str, round: int, *args, **kwargs) -> None:
        """Record the message."""
        self.received.append((source, args))


###
# Server-level buffer tests (Phase 1)
###


class TestMessageBuffer:
    """Tests for the server-level message buffer."""

    @pytest.mark.asyncio
    async def test_unknown_message_is_buffered(self):
        """Messages for unknown commands are buffered instead of dropped."""
        proto = MemoryCommunicationProtocol()
        proto.set_address("node_a")
        await proto.start()

        try:
            server = proto._server

            # Send a message for a command that doesn't exist yet
            msg = node_pb2.RootMessage(
                cmd="future_cmd",
                source="node_b",
                round=0,
            )
            msg.gossip_message.args.extend(["arg1"])
            msg.gossip_message.ttl = 5
            msg.gossip_message.hash = 12345

            await server.send(msg, None)

            assert len(server._pending_msgs_buffer) == 1
            assert server._pending_msgs_buffer[0].cmd == "future_cmd"
        finally:
            await proto.stop()

    @pytest.mark.asyncio
    async def test_buffered_message_replayed_on_add_command(self):
        """Buffered messages are replayed when their command is registered."""
        proto = MemoryCommunicationProtocol()
        proto.set_address("node_a")
        await proto.start()

        try:
            server = proto._server

            # Buffer a message
            msg = node_pb2.RootMessage(
                cmd="future_cmd",
                source="node_b",
                round=0,
            )
            msg.gossip_message.args.extend(["hello"])
            msg.gossip_message.ttl = 5
            msg.gossip_message.hash = 99999

            await server.send(msg, None)
            assert len(server._pending_msgs_buffer) == 1

            # Register the command — should trigger replay
            cmd = TrackingCommand("future_cmd")
            server.add_command(cmd)

            # Replay creates background tasks, give them a tick to run
            await asyncio.sleep(0.1)

            assert len(cmd.received) == 1
            assert cmd.received[0] == ("node_b", ("hello",))
            # Buffer should be cleared for this command
            assert len(server._pending_msgs_buffer) == 0
        finally:
            await proto.stop()

    @pytest.mark.asyncio
    async def test_buffer_retains_unmatched_messages(self):
        """Messages for still-unknown commands stay in the buffer after partial replay."""
        proto = MemoryCommunicationProtocol()
        proto.set_address("node_a")
        await proto.start()

        try:
            server = proto._server

            # Buffer two messages for different commands
            for cmd_name, hash_val in [("cmd_a", 111), ("cmd_b", 222)]:
                msg = node_pb2.RootMessage(cmd=cmd_name, source="node_b", round=0)
                msg.gossip_message.args.extend(["data"])
                msg.gossip_message.ttl = 3
                msg.gossip_message.hash = hash_val
                await server.send(msg, None)

            assert len(server._pending_msgs_buffer) == 2

            # Register only cmd_a
            cmd_a = TrackingCommand("cmd_a")
            server.add_command(cmd_a)
            await asyncio.sleep(0.1)

            assert len(cmd_a.received) == 1
            # cmd_b should still be buffered
            assert len(server._pending_msgs_buffer) == 1
            assert server._pending_msgs_buffer[0].cmd == "cmd_b"
        finally:
            await proto.stop()

    @pytest.mark.asyncio
    async def test_remove_command_clears_buffer(self):
        """remove_command() clears buffered messages for the removed command."""
        proto = MemoryCommunicationProtocol()
        proto.set_address("node_a")
        await proto.start()

        try:
            server = proto._server

            msg = node_pb2.RootMessage(cmd="stale_cmd", source="node_b", round=0)
            msg.gossip_message.args.extend(["data"])
            msg.gossip_message.ttl = 3
            msg.gossip_message.hash = 333
            await server.send(msg, None)

            assert len(server._pending_msgs_buffer) == 1

            server.remove_command("stale_cmd")
            assert len(server._pending_msgs_buffer) == 0
        finally:
            await proto.stop()

    @pytest.mark.asyncio
    async def test_buffer_max_size(self):
        """Buffer respects the max size limit."""
        proto = MemoryCommunicationProtocol()
        proto.set_address("node_a")
        await proto.start()

        try:
            server = proto._server
            max_size = Settings.general.MSG_BUFFER_SIZE

            # Fill buffer beyond max
            for i in range(max_size + 10):
                msg = node_pb2.RootMessage(cmd="overflow_cmd", source="node_b", round=0)
                msg.gossip_message.args.extend([str(i)])
                msg.gossip_message.ttl = 3
                msg.gossip_message.hash = 10000 + i
                await server.send(msg, None)

            assert len(server._pending_msgs_buffer) == max_size
        finally:
            await proto.stop()


###
# End-to-end: line topology with deferred replay (Phase 2)
###


class TestMessageBufferingE2E:
    """End-to-end test: messages arriving before start_learning are buffered and replayed."""

    @pytest.mark.asyncio
    @pytest.mark.e2e_train
    async def test_line_topology_buffered_hello(self):
        """
        In a line topology, all nodes receive hello messages despite gossip propagation delay.

        node_0 starts learning first and broadcasts hello. Distant nodes may receive
        hello before start_learning reaches them. The buffer + replay-after-RUNNING
        fix ensures no messages are lost.
        """

        # Minimal workflow: broadcast hello, wait for all peers
        class HelloStage(Stage[WorkflowContext]):
            async def run(self) -> str | None:
                ctx = self.ctx
                self._peers_ready = asyncio.Event()
                self._known_peers: set[str] = {ctx.address}

                await ctx.cp.broadcast_gossip(ctx.cp.build_msg("test_hello"))

                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._peers_ready.wait(), timeout=15)
                return None

            @on_message("test_hello")
            async def handle_hello(self, source: str, round: int, *args) -> None:
                self._known_peers.add(source)
                expected = len(self.ctx.cp.get_neighbors(only_direct=False)) + 1
                if len(self._known_peers) >= expected:
                    self._peers_ready.set()

        class HelloWorkflow(Workflow[WorkflowContext]):
            context_class = WorkflowContext

            def get_stages(self):
                return [HelloStage()]

        register_workflow("test_hello_wf", HelloWorkflow)

        # Use fast gossip for the test
        Settings.gossip.PERIOD = 1
        Settings.training.SYNCHRONIZATION_TIMEOUT = 15

        from p2pfl.examples.mnist.model.mlp_pytorch import model_build_fn

        data = P2PFLDataset.from_huggingface("p2pfl/MNIST")
        data.set_batch_size(128)
        partitions = data.generate_partitions(4, RandomIIDPartitionStrategy)

        nodes = []
        for i in range(4):
            node = Node(
                model_build_fn(),
                partitions[i],
                protocol=MemoryCommunicationProtocol(),
            )
            await node.start()
            nodes.append(node)

        try:
            # Line topology: node -- node_1 -- node_2 -- node_3
            adj = TopologyFactory.generate_matrix(TopologyType.LINE, 4)
            await TopologyFactory.connect_nodes(adj, nodes)
            await wait_convergence(nodes, 3, only_direct=False, wait=30)

            # Start learning on node 0
            await nodes[0].set_start_learning(rounds=1, epochs=0, workflow="test_hello_wf")

            # All nodes should finish without timeout
            await wait_to_finish(nodes, timeout=20)

            # Verify all nodes finished successfully
            for node in nodes:
                assert not node.state.is_learning, f"{node.address} is still learning"
                assert node.state.is_terminal or node.state == node.state.IDLE, f"{node.address} state: {node.state}"
        finally:
            for n in nodes:
                await n.stop()
