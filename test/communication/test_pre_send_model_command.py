#
# This file is part of the federated_learning_p2p (p2pfl) distribution
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

"""Tests for pre_send_model message handling in workflows."""

from unittest.mock import MagicMock

import pytest

from p2pfl.workflow.basic_dfl.workflow import BasicDFL, BasicPeerState
from p2pfl.workflow.engine.experiment import Experiment


@pytest.fixture
def mock_node():
    """Create a mock node for testing."""
    node = MagicMock()
    node.address = "test_node"
    return node


@pytest.fixture
def workflow(mock_node):
    """Create a workflow for testing."""
    return BasicDFL(node=mock_node)


def set_round(workflow, round_num):
    """Set up experiment with a specific round."""
    workflow.experiment = Experiment("test_exp", 10)
    if round_num is not None:
        workflow.round = round_num


class TestPreSendModelHandler:
    """Tests for handle_pre_send_model handler."""

    @pytest.mark.asyncio
    async def test_missing_args_returns_false(self, workflow):
        """Test that missing arguments returns false."""
        result = await workflow.handle_pre_send_model("source", 1)
        assert result == "false"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "local_round,incoming_round,expected",
        [
            (5, 6, "true"),  # Newer round accepted
            (5, 4, "false"),  # Older round rejected
            (5, 5, "false"),  # Same round rejected
        ],
    )
    async def test_add_model_round_logic(self, workflow, local_round, incoming_round, expected):
        """Test that add_model accepts only newer rounds."""
        set_round(workflow, local_round)
        result = await workflow.handle_pre_send_model("source", incoming_round, "add_model")
        assert result == expected

    @pytest.mark.asyncio
    async def test_add_model_accepts_when_round_is_zero(self, workflow):
        """Test that add_model accepts when round is 0 (no experiment set)."""
        # No experiment set, round defaults to 0
        result = await workflow.handle_pre_send_model("source", 1, "add_model")
        assert result == "true"

    @pytest.mark.asyncio
    async def test_partial_model_accepts_new_contributors(self, workflow):
        """Test that partial_model accepts new contributors."""
        # No existing contributors
        result = await workflow.handle_pre_send_model("source", 1, "partial_model", "new_contrib")
        assert result == "true"

    @pytest.mark.asyncio
    async def test_partial_model_rejects_existing_contributors(self, workflow):
        """Test that partial_model rejects existing contributors."""
        # Add existing contributor via a model
        mock_model = MagicMock()
        mock_model.get_contributors.return_value = ["contrib1"]
        workflow.peers["contrib1"] = BasicPeerState()
        workflow.peers["contrib1"].model = mock_model

        result = await workflow.handle_pre_send_model("source", 1, "partial_model", "contrib1")
        assert result == "false"

    @pytest.mark.asyncio
    async def test_partial_model_accepts_partial_new_contributors(self, workflow):
        """Test that partial_model accepts if any contributor is new."""
        # Add existing contributor
        mock_model = MagicMock()
        mock_model.get_contributors.return_value = ["contrib1"]
        workflow.peers["contrib1"] = BasicPeerState()
        workflow.peers["contrib1"].model = mock_model

        result = await workflow.handle_pre_send_model("source", 1, "partial_model", "contrib1", "new_contrib")
        assert result == "true"

    @pytest.mark.asyncio
    async def test_partial_model_empty_contributors_rejected(self, workflow):
        """Test that partial_model with no contributors is rejected."""
        result = await workflow.handle_pre_send_model("source", 1, "partial_model")
        assert result == "false"

    @pytest.mark.asyncio
    async def test_unknown_command_accepts(self, workflow):
        """Test that unknown commands are accepted by default."""
        result = await workflow.handle_pre_send_model("source", 1, "unknown_command", "contrib1")
        assert result == "true"


class TestPreSendModelIntegration:
    """Integration tests for pre_send_model flow."""

    @pytest.mark.asyncio
    async def test_workflow_state_changes_affect_decision(self, workflow):
        """Test that workflow state changes affect accept/reject decisions."""
        # First model from contrib1 - should be accepted
        result = await workflow.handle_pre_send_model("source", 1, "partial_model", "contrib1")
        assert result == "true"

        # Simulate receiving the model
        mock_model = MagicMock()
        mock_model.get_contributors.return_value = ["contrib1"]
        workflow.peers["contrib1"] = BasicPeerState()
        workflow.peers["contrib1"].model = mock_model

        # Second model from same contributor - should be rejected
        result = await workflow.handle_pre_send_model("source", 1, "partial_model", "contrib1")
        assert result == "false"

        # Model from new contributor - should be accepted
        result = await workflow.handle_pre_send_model("source", 1, "partial_model", "contrib2")
        assert result == "true"
