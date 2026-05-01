"""Tests for ProtobuffServer missed coverage paths."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from p2pfl.communication.commands.command import Command
from p2pfl.communication.protocols.protobuff.memory import MemoryCommunicationProtocol
from p2pfl.communication.protocols.protobuff.proto import node_pb2
from p2pfl.utils.utils import set_standalone_settings

set_standalone_settings()


class _TrackingCommand(Command):
    """Command that records every invocation."""

    def __init__(self, name: str = "test_cmd") -> None:
        """Initialize tracking command."""
        self._name = name
        self.calls: list[tuple] = []

    def get_name(self) -> str:  # type: ignore[override]
        """Get the command name."""
        return self._name

    async def execute(self, source: str, round: int, *args, **kwargs) -> str | None:
        """Record the call and return None."""
        self.calls.append((source, round, args, kwargs))
        return None


class _FailingCommand(Command):
    """Command that raises on execution."""

    def __init__(self, name: str = "fail_cmd") -> None:
        """Initialize failing command."""
        self._name = name

    def get_name(self) -> str:  # type: ignore[override]
        """Get the command name."""
        return self._name

    async def execute(self, *args, **kwargs) -> None:
        """Raise ValueError on execution."""
        raise ValueError("boom")


# ========== Handshake error path ==========


class TestHandshakeError:
    """Handshake error path tests."""

    @pytest.mark.asyncio
    async def test_handshake_returns_error_on_duplicate(self):
        """handshake() returns error when the neighbor cannot be added."""
        proto = MemoryCommunicationProtocol()
        proto.set_address("node_a")
        await proto.start()
        try:
            server = proto._server
            # Mock neighbors.add to return False (duplicate/invalid)
            server._neighbors.add = AsyncMock(return_value=False)
            res = await server.handshake(
                node_pb2.HandShakeRequest(addr="node_b"),
                None,  # type: ignore
            )
            assert res.error != ""
            assert "Cannot add" in res.error
        finally:
            await proto.stop()


# ========== send() edge cases ==========


class TestServerSend:
    """Server send edge case tests."""

    @pytest.mark.asyncio
    async def test_send_no_payload_field_returns_error(self):
        """send() with a RootMessage that has no payload field returns error."""
        proto = MemoryCommunicationProtocol()
        proto.set_address("node_a")
        await proto.start()
        try:
            server = proto._server
            cmd = _TrackingCommand("some_cmd")
            server.add_command(cmd)

            # Build message with no gossip_message, direct_message, or weights
            msg = node_pb2.RootMessage(cmd="some_cmd", source="node_b", round=0)
            res = await server.send(msg, None)
            assert res.error != ""
            assert "No message or weights" in res.error
        finally:
            await proto.stop()

    @pytest.mark.asyncio
    async def test_send_command_exception_returns_error(self):
        """send() catches exceptions from command execution and returns error."""
        proto = MemoryCommunicationProtocol()
        proto.set_address("node_a")
        await proto.start()
        try:
            server = proto._server
            cmd = _FailingCommand("fail_cmd")
            server.add_command(cmd)

            # Use direct_message since it's awaited (exception caught synchronously)
            msg = node_pb2.RootMessage(cmd="fail_cmd", source="node_b", round=0)
            msg.direct_message.args.extend(["arg1"])
            res = await server.send(msg, None)
            assert res.error != ""
            assert "ValueError" in res.error
        finally:
            await proto.stop()

    @pytest.mark.asyncio
    async def test_send_weights_message_executes_command(self):
        """send() with a weights payload dispatches to the command as a background task."""
        proto = MemoryCommunicationProtocol()
        proto.set_address("node_a")
        await proto.start()
        try:
            server = proto._server
            cmd = _TrackingCommand("weight_cmd")
            server.add_command(cmd)

            msg = node_pb2.RootMessage(cmd="weight_cmd", source="node_b", round=1)
            msg.weights.weights = b"\xaa\xbb"
            msg.weights.contributors.extend(["node_b"])
            msg.weights.num_samples = 50

            res = await server.send(msg, None)
            assert res.error == ""
            await asyncio.sleep(0.05)

            assert len(cmd.calls) == 1
            assert cmd.calls[0][3]["weights"] == b"\xaa\xbb"
            assert cmd.calls[0][3]["num_samples"] == 50
        finally:
            await proto.stop()

    @pytest.mark.asyncio
    async def test_send_direct_message_returns_response(self):
        """send() with a direct_message returns the command's response."""
        proto = MemoryCommunicationProtocol()
        proto.set_address("node_a")
        await proto.start()
        try:
            server = proto._server

            class _ReplyCommand(Command):
                """Command that returns a reply string."""

                def get_name(self) -> str:  # type: ignore[override]
                    """Get the command name."""
                    return "reply_cmd"

                async def execute(self, *args, **kwargs) -> str:
                    """Return a reply string."""
                    return "hello_back"

            cmd = _ReplyCommand()
            server.add_command(cmd)

            msg = node_pb2.RootMessage(cmd="reply_cmd", source="node_b", round=0)
            msg.direct_message.args.extend(["hi"])
            res = await server.send(msg, None)
            assert res.error == ""
            assert res.response == "hello_back"
        finally:
            await proto.stop()


# ========== Background task exception logging ==========


class TestBackgroundTaskTracking:
    """Background task tracking tests."""

    @pytest.mark.asyncio
    async def test_background_task_exception_is_logged(self):
        """_track_background_task logs exceptions from failed tasks."""
        proto = MemoryCommunicationProtocol()
        proto.set_address("node_a")
        await proto.start()
        try:
            server = proto._server

            async def _fail():
                raise RuntimeError("bg fail")

            task = asyncio.create_task(_fail())
            server._track_background_task(task, "test_bg_cmd")
            # Wait for task to finish
            await asyncio.sleep(0.05)
            # Task should have been discarded from the set
            assert task not in server._background_tasks
        finally:
            await proto.stop()


# ========== add_command invalid type ==========


class TestAddCommandInvalid:
    """Invalid add_command argument tests."""

    @pytest.mark.asyncio
    async def test_add_command_invalid_type_raises(self):
        """add_command raises for invalid argument types."""
        proto = MemoryCommunicationProtocol()
        proto.set_address("node_a")
        await proto.start()
        try:
            server = proto._server
            with pytest.raises(Exception, match="not valid"):
                server.add_command(42)  # type: ignore
        finally:
            await proto.stop()


# ========== Replay buffer: weights and direct_message ==========


class TestReplayBuffer:
    """Replay buffer tests for weights and direct messages."""

    @pytest.mark.asyncio
    async def test_replay_buffered_weights(self):
        """Buffered weights messages are replayed on add_command."""
        proto = MemoryCommunicationProtocol()
        proto.set_address("node_a")
        await proto.start()
        try:
            server = proto._server

            # Buffer a weights message
            msg = node_pb2.RootMessage(cmd="weight_cmd", source="node_b", round=1)
            msg.weights.weights = b"\x01\x02\x03"
            msg.weights.contributors.extend(["node_b"])
            msg.weights.num_samples = 100

            await server.send(msg, None)
            assert len(server._pending_msgs_buffer) == 1

            cmd = _TrackingCommand("weight_cmd")
            server.add_command(cmd)
            await asyncio.sleep(0.1)

            assert len(cmd.calls) == 1
            assert cmd.calls[0][3]["weights"] == b"\x01\x02\x03"
            assert cmd.calls[0][3]["num_samples"] == 100
            assert len(server._pending_msgs_buffer) == 0
        finally:
            await proto.stop()

    @pytest.mark.asyncio
    async def test_replay_buffered_direct_message(self):
        """Buffered direct messages are replayed on add_command."""
        proto = MemoryCommunicationProtocol()
        proto.set_address("node_a")
        await proto.start()
        try:
            server = proto._server

            msg = node_pb2.RootMessage(cmd="direct_cmd", source="node_b", round=0)
            msg.direct_message.args.extend(["arg1", "arg2"])

            await server.send(msg, None)
            assert len(server._pending_msgs_buffer) == 1

            cmd = _TrackingCommand("direct_cmd")
            server.add_command(cmd)
            await asyncio.sleep(0.1)

            assert len(cmd.calls) == 1
            assert cmd.calls[0][2] == ("arg1", "arg2")
            assert len(server._pending_msgs_buffer) == 0
        finally:
            await proto.stop()


# ========== remove_command ==========


class TestRemoveCommand:
    """Remove command tests."""

    @pytest.mark.asyncio
    async def test_remove_command_by_str(self):
        """remove_command accepts a string name."""
        proto = MemoryCommunicationProtocol()
        proto.set_address("node_a")
        await proto.start()
        try:
            server = proto._server
            cmd = _TrackingCommand("rm_str")
            server.add_command(cmd)
            server.remove_command("rm_str")
            # Should not raise, command is gone
        finally:
            await proto.stop()

    @pytest.mark.asyncio
    async def test_remove_command_by_instance(self):
        """remove_command accepts a Command instance."""
        proto = MemoryCommunicationProtocol()
        proto.set_address("node_a")
        await proto.start()
        try:
            server = proto._server
            cmd = _TrackingCommand("rm_inst")
            server.add_command(cmd)
            server.remove_command(cmd)
        finally:
            await proto.stop()

    @pytest.mark.asyncio
    async def test_remove_command_by_list(self):
        """remove_command accepts a list of strings and Command instances."""
        proto = MemoryCommunicationProtocol()
        proto.set_address("node_a")
        await proto.start()
        try:
            server = proto._server
            cmd1 = _TrackingCommand("rm_a")
            cmd2 = _TrackingCommand("rm_b")
            server.add_command([cmd1, cmd2])
            server.remove_command(["rm_a", cmd2])
        finally:
            await proto.stop()

    @pytest.mark.asyncio
    async def test_remove_command_invalid_type_raises(self):
        """remove_command raises for invalid argument types."""
        proto = MemoryCommunicationProtocol()
        proto.set_address("node_a")
        await proto.start()
        try:
            server = proto._server
            with pytest.raises(Exception, match="not valid"):
                server.remove_command(42)  # type: ignore
        finally:
            await proto.stop()

    @pytest.mark.asyncio
    async def test_remove_command_clears_buffered_for_removed_names(self):
        """remove_command purges buffer entries matching the removed command names."""
        proto = MemoryCommunicationProtocol()
        proto.set_address("node_a")
        await proto.start()
        try:
            server = proto._server

            # Buffer messages for two commands
            for name, h in [("keep_cmd", 111), ("drop_cmd", 222)]:
                msg = node_pb2.RootMessage(cmd=name, source="node_b", round=0)
                msg.gossip_message.args.extend(["x"])
                msg.gossip_message.ttl = 1
                msg.gossip_message.hash = h
                await server.send(msg, None)

            assert len(server._pending_msgs_buffer) == 2

            server.remove_command("drop_cmd")
            assert len(server._pending_msgs_buffer) == 1
            assert server._pending_msgs_buffer[0].cmd == "keep_cmd"
        finally:
            await proto.stop()
