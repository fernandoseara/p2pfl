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
"""Tests for GrpcClient."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from p2pfl.communication.protocols.exceptions import CommunicationError, NeighborNotConnectedError
from p2pfl.communication.protocols.protobuff.grpc.client import GrpcClient
from p2pfl.communication.protocols.protobuff.proto import node_pb2


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
        """Test that connect returns early when already connected."""
        client = GrpcClient("127.0.0.1:8000", "127.0.0.1:9000")
        client.stub = MagicMock()
        client.channel = MagicMock()

        # Should return without doing anything
        await client.connect()

        # Stub should still be the mock (not replaced)
        assert client.stub is not None

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
