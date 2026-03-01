#
# This file is part of the p2pfl (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2025 Pedro Guijas Bravo.
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
"""Tests for communication commands."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from p2pfl.communication.commands.command import Command
from p2pfl.communication.commands.infrastructure.heartbeat_command import HeartbeatCommand
from p2pfl.communication.commands.infrastructure.metrics_command import MetricsCommand
from p2pfl.communication.commands.infrastructure.start_learning_command import StartLearningCommand
from p2pfl.communication.commands.infrastructure.stop_learning_command import StopLearningCommand
from p2pfl.communication.commands.workflow.workflow_command import WorkflowCommand
from p2pfl.communication.protocols.exceptions import CommunicationError, NeighborNotConnectedError
from p2pfl.communication.protocols.protobuff.gossiper import Gossiper
from p2pfl.communication.protocols.protobuff.grpc.address import AddressParser
from p2pfl.communication.protocols.protobuff.grpc.client import GrpcClient
from p2pfl.communication.protocols.protobuff.proto import node_pb2
from p2pfl.exceptions import NodeRunningException
from p2pfl.settings import Settings
from p2pfl.utils.utils import set_standalone_settings
from p2pfl.workflow.engine.experiment import Experiment
from p2pfl.workflow.engine.message import MessageEntry

set_standalone_settings()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_node_not_learning():
    """Mock node that is not learning."""
    node = MagicMock()
    node.state.is_learning = False
    node.address = "127.0.0.1:8000"
    return node


@pytest.fixture
def mock_node_learning():
    """Mock node that is learning with a mock workflow."""
    node = MagicMock()
    node.state.is_learning = True
    node.address = "127.0.0.1:8000"
    node.workflow = MagicMock()
    node.workflow._handlers = {}
    node.workflow.current_stage_name = "setup"
    return node


@pytest.fixture
def gossiper():
    """Gossiper instance with mock neighbors."""
    mock_neighbors = MagicMock()
    g = Gossiper(mock_neighbors, MagicMock())
    g.set_address("127.0.0.1:8000")
    return g


# =============================================================================
# WorkflowCommand Tests
# =============================================================================


class TestWorkflowCommand:
    """Tests for WorkflowCommand."""

    @pytest.mark.asyncio
    async def test_execute_returns_early_when_not_learning(self, mock_node_not_learning):
        """Test that execute returns early when node is not learning."""
        cmd = WorkflowCommand(mock_node_not_learning, "test_cmd")
        result = await cmd.execute("source", 1)
        assert result is None

    @pytest.mark.asyncio
    async def test_execute_returns_none_when_handler_missing(self, mock_node_learning):
        """Test that execute returns None when no handler is registered."""
        cmd = WorkflowCommand(mock_node_learning, "unknown_cmd")
        result = await cmd.execute("source", 1)

        assert result is None

    @pytest.mark.asyncio
    async def test_routes_message_args_to_handler(self, mock_node_learning):
        """Test that message args are forwarded to handler."""
        handler = AsyncMock(return_value="success")
        mock_node_learning.workflow._handlers = {"test": [(handler, MessageEntry("h", False))]}

        cmd = WorkflowCommand(mock_node_learning, "test")
        result = await cmd.execute("source", 1, "arg1", "arg2")

        handler.assert_called_once_with("source", 1, "arg1", "arg2")
        assert result == "success"

    @pytest.mark.asyncio
    async def test_routes_weights_to_handler(self, mock_node_learning):
        """Test that weight data is forwarded to handler."""
        handler = AsyncMock()
        mock_node_learning.workflow._handlers = {"partial_model": [(handler, MessageEntry("h", True))]}

        cmd = WorkflowCommand(mock_node_learning, "partial_model")
        await cmd.execute("source", 1, weights=b"model_data", contributors=["node1", "node2"], num_samples=100)

        handler.assert_called_once_with("source", 1, b"model_data", ["node1", "node2"], 100)


# =============================================================================
# MetricsCommand Tests
# =============================================================================


class TestMetricsCommand:
    """Tests for MetricsCommand."""

    @pytest.mark.asyncio
    async def test_execute_logs_single_metric(self):
        """Test that execute logs a single metric pair."""
        cmd = MetricsCommand()

        with patch("p2pfl.communication.commands.infrastructure.metrics_command.logger") as mock_logger:
            await cmd.execute("source", 1, "loss", "0.5")

            mock_logger.log_metric.assert_called_once_with("source", metric="loss", value=0.5, round=1)

    @pytest.mark.asyncio
    async def test_execute_logs_multiple_metrics(self):
        """Test that execute logs multiple metric pairs."""
        cmd = MetricsCommand()

        with patch("p2pfl.communication.commands.infrastructure.metrics_command.logger") as mock_logger:
            await cmd.execute("source", 2, "loss", "0.5", "accuracy", "0.95")

            assert mock_logger.log_metric.call_count == 2
            mock_logger.log_metric.assert_any_call("source", metric="loss", value=0.5, round=2)
            mock_logger.log_metric.assert_any_call("source", metric="accuracy", value=0.95, round=2)


# =============================================================================
# StartLearningCommand Tests
# =============================================================================


class TestStartLearningCommand:
    """Tests for StartLearningCommand."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"learning_rounds": None, "learning_epochs": 5, "workflow": "basic"},
            {"learning_rounds": 10, "learning_epochs": None, "workflow": "basic"},
            {"learning_rounds": 10, "learning_epochs": 5, "workflow": None},
        ],
    )
    async def test_execute_raises_on_missing_required_param(self, kwargs):
        """Test that execute raises ValueError when required params are missing."""
        mock_node = MagicMock()
        cmd = StartLearningCommand(mock_node)

        with pytest.raises(ValueError, match="required"):
            await cmd.execute("source", 1, trainset_size=100, experiment_name="test", **kwargs)

    @pytest.mark.asyncio
    async def test_execute_calls_start_learning_workflow(self):
        """Test that execute calls node._start_learning_workflow with correct args."""
        mock_node = MagicMock()
        mock_node._start_learning_workflow = AsyncMock()

        cmd = StartLearningCommand(mock_node)
        await cmd.execute(
            "source",
            1,
            learning_rounds="10",
            learning_epochs="5",
            experiment_name="my_experiment",
            workflow="basic",
            workflow_kwargs={"trainset_size": 100},
        )

        mock_node._start_learning_workflow.assert_called_once()
        args, kwargs = mock_node._start_learning_workflow.call_args
        assert args[0] == "basic"
        assert isinstance(args[1], Experiment)
        assert args[1].total_rounds == 10
        assert args[1].epochs_per_round == 5
        assert args[1].exp_name == "my_experiment"
        assert kwargs["trainset_size"] == 100

    @pytest.mark.asyncio
    async def test_execute_handles_node_running_exception(self):
        """Test that NodeRunningException is caught and logged."""
        mock_node = MagicMock()
        mock_node.address = "127.0.0.1:8000"
        mock_node._start_learning_workflow = AsyncMock(side_effect=NodeRunningException("Already running"))

        cmd = StartLearningCommand(mock_node)

        # Should not raise
        await cmd.execute(
            "source",
            1,
            learning_rounds=10,
            learning_epochs=5,
            trainset_size=100,
            experiment_name="test",
            workflow="basic",
        )


# =============================================================================
# StopLearningCommand Tests
# =============================================================================


class TestStopLearningCommand:
    """Tests for StopLearningCommand."""

    @pytest.mark.asyncio
    async def test_execute_stops_learning_when_active(self, mock_node_learning):
        """Test that execute stops learning when node is actively learning."""
        mock_node_learning._stop_workflow = AsyncMock()

        cmd = StopLearningCommand(mock_node_learning)
        await cmd.execute("source", 1)

        mock_node_learning._stop_workflow.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_does_nothing_when_not_learning(self, mock_node_not_learning):
        """Test that execute does nothing when node is not learning."""
        mock_node_not_learning._stop_workflow = AsyncMock()

        cmd = StopLearningCommand(mock_node_not_learning)
        await cmd.execute("source", 1)

        # StopLearningCommand checks is_learning, which is False, so _stop_workflow() should not be called
        mock_node_not_learning._stop_workflow.assert_not_called()


# =============================================================================
# HeartbeatCommand Tests
# =============================================================================


class TestHeartbeatCommand:
    """Tests for HeartbeatCommand."""

    @pytest.mark.asyncio
    async def test_execute_raises_on_missing_time(self):
        """Test that execute raises ValueError when time is None."""
        mock_heartbeater = MagicMock()
        cmd = HeartbeatCommand(mock_heartbeater)

        with pytest.raises(ValueError, match="Time is required"):
            await cmd.execute("source", 1, time=None)

    @pytest.mark.asyncio
    async def test_execute_calls_heartbeater_beat(self):
        """Test that execute calls heartbeater.beat with correct args."""
        mock_heartbeater = MagicMock()
        mock_heartbeater.beat = AsyncMock()
        cmd = HeartbeatCommand(mock_heartbeater)

        await cmd.execute("source", 1, time="1234567890.123")

        mock_heartbeater.beat.assert_called_once_with("source", time=1234567890.123)


# =============================================================================
# AddressParser Tests
# =============================================================================


class TestAddressParser:
    """Tests for AddressParser."""

    @pytest.mark.parametrize(
        "address,expected_host,expected_port,is_v6,unix_domain,parsed",
        [
            ("127.0.0.1:8080", "127.0.0.1", 8080, False, False, "127.0.0.1:8080"),
            ("[::1]:8080", "::1", 8080, True, False, "[::1]:8080"),
            ("[2001:db8::1]:9000", "2001:db8::1", 9000, True, False, "[2001:db8::1]:9000"),
            ("unix:///var/run/socket", "unix:///var/run/socket", None, None, True, "unix:///var/run/socket"),
        ],
    )
    def test_valid_address_parsing(self, address, expected_host, expected_port, is_v6, unix_domain, parsed):
        """Test parsing valid addresses (IPv4, IPv6, Unix domain)."""
        parser = AddressParser(address)

        assert parser.host == expected_host
        if expected_port is not None:
            assert parser.port == expected_port
        assert parser.is_v6 == is_v6
        assert parser.unix_domain is unix_domain
        assert parser.get_parsed_address() == parsed

    def test_unix_domain_relative_path_not_recognized(self):
        """Test that relative Unix paths are not recognized as Unix domain."""
        parser = AddressParser("unix://relative/path")
        assert parser.unix_domain is False

    @pytest.mark.parametrize("address", ["127.0.0.1:70000", "127.0.0.1:0", "127.0.0.1:99999"])
    def test_invalid_port_results_in_none(self, address):
        """Test that invalid ports result in None host/port."""
        parser = AddressParser(address)
        assert parser.host is None
        assert parser.port is None

    def test_get_parsed_address_raises_on_invalid(self):
        """Test that get_parsed_address raises ValueError for invalid address."""
        parser = AddressParser("127.0.0.1:99999")
        with pytest.raises(ValueError, match="invalid"):
            parser.get_parsed_address()

    def test_hostname_only_assigns_random_port(self):
        """Test that hostname-only address gets random port assigned."""
        parser = AddressParser("localhost")
        assert parser.host is not None
        assert parser.port is not None
        assert 1 <= parser.port <= 65535


# =============================================================================
# GrpcClient Tests
# =============================================================================


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


# =============================================================================
# Gossiper Tests
# =============================================================================


class TestGossiper:
    """Tests for Gossiper."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "msg_source,expected",
        [
            ("127.0.0.1:8000", False),  # Own message
            ("127.0.0.1:9000", True),  # New message from other
        ],
    )
    async def test_check_and_set_processed(self, gossiper, msg_source, expected):
        """Test message processing: own messages rejected, new messages accepted."""
        mock_msg = MagicMock()
        mock_msg.source = msg_source
        mock_msg.gossip_message.hash = 12345

        result = await gossiper.check_and_set_processed(mock_msg)
        assert result is expected

    @pytest.mark.asyncio
    async def test_check_and_set_processed_returns_false_for_duplicate(self, gossiper):
        """Test that duplicate messages return False."""
        mock_msg = MagicMock()
        mock_msg.source = "127.0.0.1:9000"
        mock_msg.gossip_message.hash = 12345

        assert await gossiper.check_and_set_processed(mock_msg) is True
        assert await gossiper.check_and_set_processed(mock_msg) is False

    @pytest.mark.asyncio
    async def test_add_message_queues_for_neighbors(self, gossiper):
        """Test that add_message queues message for all direct neighbors."""
        mock_client = MagicMock()
        gossiper._neighbors.get_all.return_value = {
            "127.0.0.1:9000": (mock_client, 0),
            "127.0.0.1:9001": (mock_client, 0),
        }

        mock_msg = MagicMock()
        mock_msg.source = "127.0.0.1:7000"

        await gossiper.add_message(mock_msg)

        assert len(gossiper._pending_msgs) == 1
        assert gossiper._pending_msgs[0][0] == mock_msg

    @pytest.mark.asyncio
    async def test_circular_buffer_limits_processed_messages(self, gossiper):
        """Test that processed messages list is limited to configured size."""
        limit = Settings.gossip.AMOUNT_LAST_MESSAGES_SAVED
        for i in range(limit + 5):
            mock_msg = MagicMock()
            mock_msg.source = "127.0.0.1:9000"
            mock_msg.gossip_message.hash = i

            await gossiper.check_and_set_processed(mock_msg)

        # The list should have removed old entries
        assert len(gossiper._processed_messages) <= limit + 1


# =============================================================================
# Command Base Class Tests
# =============================================================================


class TestCommandBase:
    """Tests for Command base class."""

    def test_node_property_raises_when_none(self):
        """Test that node property raises RuntimeError when node is None."""

        # Create a concrete subclass for testing
        class TestCommand(Command):
            @staticmethod
            def get_name() -> str:
                return "test"

            async def execute(self, source: str, round: int, *args, **kwargs) -> str | None:
                return None

        cmd = TestCommand(node=None)

        with pytest.raises(RuntimeError, match="requires a node"):
            _ = cmd.node

    def test_workflow_property_delegates_to_node_workflow(self):
        """Test that Command.workflow delegates to node.workflow."""

        class TestCommand(Command):
            @staticmethod
            def get_name() -> str:
                return "test"

            async def execute(self, source: str, round: int, *args, **kwargs) -> str | None:
                return None

        mock_node = MagicMock()
        mock_workflow = MagicMock()
        mock_node.workflow = mock_workflow

        cmd = TestCommand(node=mock_node)

        assert cmd.workflow == mock_workflow
