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
from p2pfl.exceptions import NodeRunningException
from p2pfl.workflow.engine.experiment import Experiment
from p2pfl.workflow.engine.message import MessageEntry

###
# Fixtures
###


@pytest.fixture
def mock_node_not_learning():
    """Mock node that is not learning."""
    node = MagicMock()
    node.state.is_learning = False
    node.address = "127.0.0.1:8000"
    node.workflow = None
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


###
# WorkflowCommand Tests
###


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


###
# MetricsCommand Tests
###


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


###
# StartLearningCommand Tests
###


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
        assert args[1].trainset_size == 100

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


###
# StopLearningCommand Tests
###


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


###
# HeartbeatCommand Tests
###


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


###
# Command Base Class Tests
###


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
