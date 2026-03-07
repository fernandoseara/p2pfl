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

"""Tests for pre_send_model_learning message handling in workflows."""

from unittest.mock import MagicMock

import pytest

from p2pfl.workflow.basic_dfl.context import BasicDFLContext, BasicPeerState
from p2pfl.workflow.basic_dfl.stages.learning import LearningStage
from p2pfl.workflow.engine.experiment import Experiment
from p2pfl.workflow.shared.gossiping import should_accept_model


@pytest.fixture
def learning_stage():
    """Create a LearningStage with a mock context for testing."""
    stage = LearningStage()
    stage.ctx = BasicDFLContext(
        address="test_node",
        learner=MagicMock(),
        aggregator=MagicMock(),
        cp=MagicMock(),
        generator=MagicMock(),
        experiment=Experiment("test_exp", 10),
    )
    return stage


def set_round(stage: LearningStage, round_num: int) -> None:
    """Set up experiment with a specific round."""
    if round_num is not None:
        stage.ctx.experiment.round = round_num


class TestShouldAcceptModel:
    """Tests for should_accept_model utility function."""

    @pytest.mark.parametrize(
        "round,local_round,expected",
        [
            (6, 5, True),  # Newer round accepted
            (4, 5, False),  # Older round rejected
            (5, 5, False),  # Same round rejected
        ],
    )
    def test_add_model_round_logic(self, round, local_round, expected):
        """Test that add_model accepts only newer rounds."""
        result = should_accept_model(
            weight_command="add_model",
            contributors=[],
            round=round,
            local_round=local_round,
            existing_contributors=set(),
        )
        assert result is expected

    def test_partial_model_accepts_new_contributors(self):
        """Test that partial_model accepts new contributors."""
        result = should_accept_model(
            weight_command="partial_model",
            contributors=["new_contrib"],
            round=1,
            local_round=1,
            existing_contributors=set(),
        )
        assert result is True

    def test_partial_model_rejects_existing_contributors(self):
        """Test that partial_model rejects existing contributors."""
        result = should_accept_model(
            weight_command="partial_model",
            contributors=["contrib1"],
            round=1,
            local_round=1,
            existing_contributors={"contrib1"},
        )
        assert result is False

    def test_partial_model_accepts_partial_new_contributors(self):
        """Test that partial_model accepts if any contributor is new."""
        result = should_accept_model(
            weight_command="partial_model",
            contributors=["contrib1", "new_contrib"],
            round=1,
            local_round=1,
            existing_contributors={"contrib1"},
        )
        assert result is True

    def test_partial_model_empty_contributors_rejected(self):
        """Test that partial_model with no contributors is rejected."""
        result = should_accept_model(
            weight_command="partial_model",
            contributors=[],
            round=1,
            local_round=1,
            existing_contributors=set(),
        )
        assert result is False

    def test_unknown_command_rejected(self):
        """Test that unknown commands are rejected by default."""
        result = should_accept_model(
            weight_command="unknown_command",
            contributors=["contrib1"],
            round=1,
            local_round=1,
            existing_contributors=set(),
        )
        assert result is False


class TestPreSendModelHandler:
    """Tests for handle_pre_send_model_learning on LearningStage."""

    @pytest.mark.asyncio
    async def test_missing_args_returns_false(self, learning_stage):
        """Test that missing arguments returns false."""
        result = await learning_stage.handle_pre_send_model_learning("source", 1)
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
    async def test_add_model_round_logic(self, learning_stage, local_round, incoming_round, expected):
        """Test that add_model accepts only newer rounds."""
        set_round(learning_stage, local_round)
        result = await learning_stage.handle_pre_send_model_learning("source", incoming_round, "add_model")
        assert result == expected

    @pytest.mark.asyncio
    async def test_add_model_accepts_when_round_is_zero(self, learning_stage):
        """Test that add_model accepts when round is 0 (no experiment set)."""
        # No round set, defaults to 0
        result = await learning_stage.handle_pre_send_model_learning("source", 1, "add_model")
        assert result == "true"

    @pytest.mark.asyncio
    async def test_partial_model_accepts_new_contributors(self, learning_stage):
        """Test that partial_model accepts new contributors."""
        result = await learning_stage.handle_pre_send_model_learning("source", 1, "partial_model", "new_contrib")
        assert result == "true"

    @pytest.mark.asyncio
    async def test_partial_model_rejects_existing_contributors(self, learning_stage):
        """Test that partial_model rejects existing contributors."""
        mock_model = MagicMock()
        mock_model.get_contributors.return_value = ["contrib1"]
        learning_stage.ctx.peers["contrib1"] = BasicPeerState()
        learning_stage.ctx.peers["contrib1"].model = mock_model

        result = await learning_stage.handle_pre_send_model_learning("source", 1, "partial_model", "contrib1")
        assert result == "false"

    @pytest.mark.asyncio
    async def test_partial_model_accepts_partial_new_contributors(self, learning_stage):
        """Test that partial_model accepts if any contributor is new."""
        mock_model = MagicMock()
        mock_model.get_contributors.return_value = ["contrib1"]
        learning_stage.ctx.peers["contrib1"] = BasicPeerState()
        learning_stage.ctx.peers["contrib1"].model = mock_model

        result = await learning_stage.handle_pre_send_model_learning("source", 1, "partial_model", "contrib1", "new_contrib")
        assert result == "true"

    @pytest.mark.asyncio
    async def test_partial_model_empty_contributors_rejected(self, learning_stage):
        """Test that partial_model with no contributors is rejected."""
        result = await learning_stage.handle_pre_send_model_learning("source", 1, "partial_model")
        assert result == "false"

    @pytest.mark.asyncio
    async def test_unknown_command_rejected(self, learning_stage):
        """Test that unknown commands are rejected by default."""
        result = await learning_stage.handle_pre_send_model_learning("source", 1, "unknown_command", "contrib1")
        assert result == "false"


class TestPreSendModelIntegration:
    """Integration tests for pre_send_model_learning flow."""

    @pytest.mark.asyncio
    async def test_workflow_state_changes_affect_decision(self, learning_stage):
        """Test that workflow state changes affect accept/reject decisions."""
        # First model from contrib1 - should be accepted
        result = await learning_stage.handle_pre_send_model_learning("source", 1, "partial_model", "contrib1")
        assert result == "true"

        # Simulate receiving the model
        mock_model = MagicMock()
        mock_model.get_contributors.return_value = ["contrib1"]
        learning_stage.ctx.peers["contrib1"] = BasicPeerState()
        learning_stage.ctx.peers["contrib1"].model = mock_model

        # Second model from same contributor - should be rejected
        result = await learning_stage.handle_pre_send_model_learning("source", 1, "partial_model", "contrib1")
        assert result == "false"

        # Model from new contributor - should be accepted
        result = await learning_stage.handle_pre_send_model_learning("source", 1, "partial_model", "contrib2")
        assert result == "true"
