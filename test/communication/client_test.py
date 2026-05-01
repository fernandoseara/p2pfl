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
"""Tests for GrpcClient and MemoryClient."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from p2pfl.communication.protocols.exceptions import CommunicationError, NeighborNotConnectedError
from p2pfl.communication.protocols.protobuff.grpc.client import GrpcClient
from p2pfl.communication.protocols.protobuff.memory.client import MemoryClient
from p2pfl.communication.protocols.protobuff.memory.singleton_dict import SingletonDict
from p2pfl.communication.protocols.protobuff.proto import node_pb2
from p2pfl.communication.protocols.protobuff.server import ProtobuffServer


def _make_response(error: str = "", response: str = "") -> node_pb2.ResponseMessage:
    return node_pb2.ResponseMessage(error=error, response=response)


def _make_gossip_msg(cmd: str = "test_cmd") -> node_pb2.RootMessage:
    msg = node_pb2.RootMessage(cmd=cmd, source="sender", round=0)
    msg.gossip_message.args.extend(["arg1"])
    msg.gossip_message.ttl = 5
    msg.gossip_message.hash = 12345
    return msg


###
# GrpcClient Tests
###


class TestGrpcClient:
    """Tests for GrpcClient."""

    @pytest.mark.asyncio
    async def test_send_raises_when_not_connected(self):
        """Test that send raises NeighborNotConnectedError when not connected."""
        client = GrpcClient("127.0.0.1:8000", "127.0.0.1:9000")
        mock_msg = MagicMock()

        with pytest.raises(NeighborNotConnectedError):
            await client.send(mock_msg, temporal_connection=False, raise_error=True)

    @pytest.mark.asyncio
    async def test_send_uses_temporal_connection(self):
        """Test that send creates temporal connection when requested."""
        client = GrpcClient("127.0.0.1:8000", "127.0.0.1:9000")

        # Create a real message instead of MagicMock to avoid serialization issues
        mock_msg = node_pb2.RootMessage(cmd="test", source="127.0.0.1:8000", round=-1)

        # Mock connect to set up stub
        async def mock_connect(handshake_msg=True):
            client.stub = MagicMock()
            client.channel = MagicMock()
            # Mock stub.send as async - return a proper response
            mock_response = MagicMock()
            mock_response.error = ""
            mock_response.response = "ok"
            client.stub.send = AsyncMock(return_value=mock_response)

        client.connect = mock_connect

        # Mock disconnect
        client.disconnect = AsyncMock()

        result = await client.send(mock_msg, temporal_connection=True)

        assert result == "ok"
        client.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_does_nothing_when_not_connected(self):
        """Test that disconnect does nothing when not connected."""
        client = GrpcClient("127.0.0.1:8000", "127.0.0.1:9000")

        # Should not raise
        await client.disconnect()

        assert client.stub is None
        assert client.channel is None

    @pytest.mark.asyncio
    async def test_connect_does_nothing_when_already_connected(self):
        """Calling connect() when already connected keeps the existing stub unchanged."""
        client = GrpcClient("127.0.0.1:8000", "127.0.0.1:9000")
        original_stub = MagicMock()
        client.stub = original_stub
        client.channel = MagicMock()

        await client.connect()

        assert client.stub is original_stub

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "raise_error,expected_result,should_raise",
        [
            (False, "", False),
            (True, None, True),
        ],
    )
    async def test_send_error_handling(self, raise_error, expected_result, should_raise):
        """Test send behavior on error with different raise_error settings."""
        client = GrpcClient("127.0.0.1:8000", "127.0.0.1:9000")
        client.stub = MagicMock()
        client.channel = MagicMock()
        client.stub.send = AsyncMock(side_effect=Exception("Connection failed"))

        mock_msg = node_pb2.RootMessage(cmd="test", source="127.0.0.1:8000", round=-1)

        if should_raise:
            with pytest.raises(Exception, match="Connection failed"):
                await client.send(mock_msg, raise_error=raise_error)
        else:
            result = await client.send(mock_msg, raise_error=raise_error)
            assert result == expected_result

    @pytest.mark.asyncio
    async def test_send_raises_communication_error_on_response_error(self):
        """Test that send raises CommunicationError when response has error."""
        client = GrpcClient("127.0.0.1:8000", "127.0.0.1:9000")
        client.stub = MagicMock()
        client.channel = MagicMock()

        # Return a response with an error
        mock_response = MagicMock()
        mock_response.error = "Command not found"
        mock_response.response = ""
        client.stub.send = AsyncMock(return_value=mock_response)

        mock_msg = node_pb2.RootMessage(cmd="unknown", source="127.0.0.1:8000", round=-1)

        with pytest.raises(CommunicationError, match="Command not found"):
            await client.send(mock_msg, raise_error=True, disconnect_on_error=False)


###
# MemoryClient — connect() edge cases
###


class TestMemoryClientConnect:
    """Memory Client Connect tests."""

    @pytest.mark.asyncio
    async def test_connect_already_connected_is_noop(self):
        """connect() on already-connected client keeps the existing stub unchanged."""
        client = MemoryClient("node_a", "node_b")
        original_stub = MagicMock(spec=ProtobuffServer)
        client.stub = original_stub
        await client.connect(handshake_msg=False)
        assert client.stub is original_stub

    @pytest.mark.asyncio
    async def test_connect_handshake_error_raises_and_clears_stub(self):
        """connect() when handshake returns an error raises and clears stub."""
        client = MemoryClient("node_a", "node_b")

        mock_server = MagicMock(spec=ProtobuffServer)
        mock_server.handshake = AsyncMock(return_value=_make_response(error="duplicate"))

        sd = SingletonDict()
        sd["node_b"] = mock_server
        try:
            with pytest.raises(Exception, match="Cannot add a neighbor"):
                await client.connect(handshake_msg=True)
            assert client.stub is None
        finally:
            sd.pop("node_b", None)

    @pytest.mark.asyncio
    async def test_connect_neighbor_not_found_raises(self):
        """connect() when neighbor not in SingletonDict raises NeighborNotConnectedError."""
        client = MemoryClient("node_a", "nonexistent_node")
        sd = SingletonDict()
        sd.pop("nonexistent_node", None)
        with pytest.raises(NeighborNotConnectedError, match="not found"):
            await client.connect()
        assert client.stub is None

    @pytest.mark.asyncio
    async def test_connect_generic_exception_clears_stub(self):
        """connect() on unexpected exception clears stub and re-raises."""
        client = MemoryClient("node_a", "node_b")

        mock_server = MagicMock(spec=ProtobuffServer)
        mock_server.handshake = AsyncMock(side_effect=OSError("network down"))

        sd = SingletonDict()
        sd["node_b"] = mock_server
        try:
            with pytest.raises(OSError, match="network down"):
                await client.connect(handshake_msg=True)
            assert client.stub is None
        finally:
            sd.pop("node_b", None)


###
# MemoryClient — disconnect() edge cases
###


class TestMemoryClientDisconnect:
    """Memory Client Disconnect tests."""

    @pytest.mark.asyncio
    async def test_disconnect_not_connected_is_noop(self):
        """disconnect() on non-connected client logs and returns."""
        client = MemoryClient("node_a", "node_b")
        assert client.stub is None
        # Should not raise
        await client.disconnect()
        assert client.stub is None

    @pytest.mark.asyncio
    async def test_disconnect_suppresses_exceptions(self):
        """disconnect() swallows exceptions from the disconnect message."""
        client = MemoryClient("node_a", "node_b")
        mock_server = MagicMock(spec=ProtobuffServer)
        mock_server.disconnect = AsyncMock(side_effect=RuntimeError("gone"))
        client.stub = mock_server
        # Should not raise despite the exception
        await client.disconnect(disconnect_msg=True)
        assert client.stub is None


###
# MemoryClient — send() edge cases
###


class TestMemoryClientSend:
    """Memory Client Send tests."""

    @pytest.mark.asyncio
    async def test_send_not_connected_temporal_connection(self):
        """send() with temporal_connection=True connects, sends, then disconnects."""
        client = MemoryClient("node_a", "node_b")

        mock_server = MagicMock(spec=ProtobuffServer)
        mock_server.handshake = AsyncMock(return_value=_make_response())
        mock_server.send = AsyncMock(return_value=_make_response())
        mock_server.disconnect = AsyncMock(return_value=None)

        sd = SingletonDict()
        sd["node_b"] = mock_server
        try:
            msg = _make_gossip_msg()
            result = await client.send(msg, temporal_connection=True)
            assert result == ""
            # After temporal connection, stub should be cleared
            assert client.stub is None
        finally:
            sd.pop("node_b", None)

    @pytest.mark.asyncio
    async def test_send_not_connected_raise_error(self):
        """send() raises NeighborNotConnectedError when not connected and raise_error=True."""
        client = MemoryClient("node_a", "node_b")
        msg = _make_gossip_msg()
        with pytest.raises(NeighborNotConnectedError):
            await client.send(msg, raise_error=True)

    @pytest.mark.asyncio
    async def test_send_not_connected_no_raise(self):
        """send() raises NeighborNotConnectedError even without raise_error when not connected."""
        client = MemoryClient("node_a", "node_b")
        msg = _make_gossip_msg()
        with pytest.raises(NeighborNotConnectedError):
            await client.send(msg, raise_error=False)

    @pytest.mark.asyncio
    async def test_send_error_response_disconnects(self):
        """send() disconnects on error response when disconnect_on_error is True."""
        client = MemoryClient("node_a", "node_b")
        mock_server = MagicMock(spec=ProtobuffServer)
        mock_server.send = AsyncMock(return_value=_make_response(error="bad request"))
        mock_server.disconnect = AsyncMock(return_value=None)
        client.stub = mock_server

        msg = _make_gossip_msg()
        await client.send(msg, disconnect_on_error=True)
        # Should have disconnected
        assert client.stub is None

    @pytest.mark.asyncio
    async def test_send_error_response_with_raise_error(self):
        """send() raises CommunicationError when response has error and raise_error=True."""
        client = MemoryClient("node_a", "node_b")
        mock_server = MagicMock(spec=ProtobuffServer)
        mock_server.send = AsyncMock(return_value=_make_response(error="server error"))
        mock_server.disconnect = AsyncMock(return_value=None)
        client.stub = mock_server

        msg = _make_gossip_msg()
        with pytest.raises(CommunicationError, match="server error"):
            await client.send(msg, raise_error=True, disconnect_on_error=True)

    @pytest.mark.asyncio
    async def test_send_temporal_connection_counter_prevents_early_disconnect(self):
        """When _temporal_connection_uses > 1, temporal send decrements but does not disconnect."""
        client = MemoryClient("node_a", "node_b")
        mock_server = MagicMock(spec=ProtobuffServer)
        mock_server.send = AsyncMock(return_value=_make_response())
        mock_server.disconnect = AsyncMock(return_value=None)

        # Simulate two callers holding temporal connections:
        # the client is already connected with uses=2
        client.stub = mock_server
        client._temporal_connection_uses = 2

        msg = _make_gossip_msg()
        await client.send(msg, temporal_connection=True)
        # Decremented from 2 to 1, should NOT disconnect
        assert client._temporal_connection_uses == 1
        assert client.stub is not None

        # Second call: decrements from 1 to 0, SHOULD disconnect
        await client.send(msg, temporal_connection=True)
        assert client._temporal_connection_uses == 0
        assert client.stub is None
