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

"""Tests for BasicDFL (new engine)."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from p2pfl.learning.frameworks.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.workflow.basic_dfl.context import BasicDFLContext, BasicPeerState
from p2pfl.workflow.basic_dfl.stages.learning_wait_model import LearningWaitModelStage
from p2pfl.workflow.basic_dfl.workflow import BasicDFL
from p2pfl.workflow.engine.experiment import Experiment
from p2pfl.workflow.engine.workflow import WorkflowStatus
from p2pfl.workflow.validation import validate


@pytest.fixture
def workflow():
    """Create a workflow for testing."""
    return BasicDFL()


@pytest.fixture
def ctx():
    """Create a typed BasicDFLContext for testing."""
    cp = MagicMock()
    return BasicDFLContext(
        address="test_node_address",
        learner=MagicMock(),
        aggregator=MagicMock(),
        cp=cp,
        generator=MagicMock(),
        experiment=Experiment.create(exp_name="test_exp", total_rounds=5, trainset_size=3),
    )


@pytest.fixture
def composed_workflow(workflow, ctx):
    """Create a workflow with stages composed and ctx wired."""
    workflow._compose(ctx)
    return workflow


class TestBasicWorkflowCreation:
    """Tests for BasicDFL creation and initialization."""

    def test_init_workflow_initial_status(self):
        """Test that initialized workflow starts with IDLE status."""
        wf = BasicDFL()
        assert wf.status == WorkflowStatus.IDLE

    def test_create_context(self):
        """Test that create_context builds a BasicDFLContext."""
        wf = BasicDFL()
        ctx = wf.create_context(
            address="test",
            learner=MagicMock(),
            aggregator=MagicMock(),
            cp=MagicMock(),
            generator=MagicMock(),
            experiment=Experiment("test", total_rounds=5),
        )
        assert isinstance(ctx, BasicDFLContext)
        assert ctx.address == "test"
        assert ctx.peers == {}
        assert ctx.train_set == []
        assert ctx.needs_full_model is False

    def test_factory_creates_basic(self):
        """Test that factory creates BasicDFL for BASIC type."""
        from p2pfl.workflow.factory import create_workflow

        wf = create_workflow("basic")
        assert isinstance(wf, BasicDFL)


class TestBasicStageMap:
    """Tests for stage map configuration."""

    def test_stage_map_has_all_stages(self):
        """Test that get_stages returns all expected stages."""
        wf = BasicDFL()
        stages = wf.get_stages()
        expected = {
            "setup",
            "round_init",
            "voting",
            "learning_evaluate",
            "learning_train",
            "learning_wait_model",
            "learning_gossip_loop",
            "learning_aggregate",
            "finish",
        }
        assert {s.name for s in stages} == expected

    def test_stage_map_types(self):
        """Test that each stage is the correct type."""
        from p2pfl.workflow.basic_dfl.stages import (
            FinishStage,
            LearningAggregateStage,
            LearningEvaluateStage,
            LearningGossipLoopStage,
            LearningTrainStage,
            LearningWaitModelStage,
            RoundInitStage,
            SetupStage,
            VotingStage,
        )

        wf = BasicDFL()
        stages = {s.name: s for s in wf.get_stages()}

        assert isinstance(stages["setup"], SetupStage)
        assert isinstance(stages["round_init"], RoundInitStage)
        assert isinstance(stages["voting"], VotingStage)
        assert isinstance(stages["learning_evaluate"], LearningEvaluateStage)
        assert isinstance(stages["learning_train"], LearningTrainStage)
        assert isinstance(stages["learning_wait_model"], LearningWaitModelStage)
        assert isinstance(stages["learning_gossip_loop"], LearningGossipLoopStage)
        assert isinstance(stages["learning_aggregate"], LearningAggregateStage)
        assert isinstance(stages["finish"], FinishStage)

    def test_stages_have_ctx_reference(self, composed_workflow, ctx):
        """Test that all stages have a reference to the ctx after composition."""
        for stage in composed_workflow._stage_map.values():
            assert stage.ctx is ctx

    def test_initial_stage(self):
        """Test that initial_stage is derived from the first stage."""
        wf = BasicDFL()
        assert wf.initial_stage == wf.get_stages()[0].name


class TestBasicDeclaredMessages:
    """Tests for declared messages (before run)."""

    def test_declared_messages_contain_all(self):
        """Test that get_messages returns all expected messages."""
        wf = BasicDFL()
        msgs = wf.get_messages()
        expected = {
            "node_initialized",
            "pre_send_initial_model",
            "initial_model",
            "node_ready",
            "peer_round_updated",
            "add_model",
            "vote_train_set",
            "models_aggregated",
            "pre_send_model_init",
            "pre_send_model_learning",
            "partial_model",
        }
        assert set(msgs.keys()) == expected

    def test_weights_messages_flagged(self):
        """Test that weight messages are properly flagged."""
        wf = BasicDFL()
        msgs = wf.get_messages()
        assert msgs["add_model"].is_weights is True
        assert msgs["partial_model"].is_weights is True
        assert msgs["initial_model"].is_weights is True
        assert msgs["node_initialized"].is_weights is False
        assert msgs["vote_train_set"].is_weights is False

    def test_during_filters_set(self):
        """
        Test that during filters are set on all handlers.

        Pre-compose, handlers without explicit ``during`` have ``during=None``
        (the owning-stage default is applied during ``_compose``).
        """
        wf = BasicDFL()
        msgs = wf.get_messages()
        # Handlers without explicit during= → None pre-compose
        assert msgs["node_initialized"].during is None
        assert msgs["initial_model"].during is None
        assert msgs["node_ready"].during is None
        assert msgs["add_model"].during == frozenset({"learning_wait_model"})
        assert msgs["models_aggregated"].during == frozenset({"learning_.*", "voting", "round_init"})
        assert msgs["pre_send_initial_model"].during == frozenset({"setup", "round_init"})
        assert msgs["pre_send_model_init"].during == frozenset({"setup", "round_init", "learning_.*", "voting"})
        assert msgs["pre_send_model_learning"].during == frozenset({"learning_.*", "voting", "round_init"})
        assert msgs["partial_model"].during == frozenset({"learning_.*", "voting", "round_init"})
        # Handlers with explicit during=
        assert msgs["peer_round_updated"].during == frozenset({"setup", "round_init", "learning_.*", "voting"})
        assert msgs["vote_train_set"].during == frozenset({"voting", "round_init", "learning_.*"})


class TestBasicMessageRegistry:
    """Tests for message registry (after composition)."""

    def test_registry_contains_all_messages(self, composed_workflow):
        """Test that the message registry contains all expected messages."""
        registry = composed_workflow.get_messages()
        expected_messages = {
            "node_initialized",
            "pre_send_initial_model",
            "initial_model",
            "node_ready",
            "peer_round_updated",
            "add_model",
            "vote_train_set",
            "models_aggregated",
            "pre_send_model_init",
            "pre_send_model_learning",
            "partial_model",
        }
        assert set(registry.keys()) == expected_messages

    def test_weights_messages_flagged(self, composed_workflow):
        """Test that weight messages are properly flagged."""
        registry = composed_workflow.get_messages()
        assert registry["add_model"].is_weights is True
        assert registry["partial_model"].is_weights is True
        assert registry["node_initialized"].is_weights is False
        assert registry["vote_train_set"].is_weights is False


class TestBasicPeerState:
    """Tests for peer state operations."""

    def test_reset_round(self):
        """Test reset_round clears per-round state."""
        peer = BasicPeerState()
        peer.model = MagicMock()
        peer.aggregated_from = ["a", "b"]
        peer.votes = {"x": 1}
        peer.reset_round()
        assert peer.model is None
        assert peer.aggregated_from == []
        assert peer.votes == {}


class TestBasicWorkflowStatus:
    """Tests for the workflow status property."""

    def test_status_idle_initially(self, workflow):
        """Test that status is IDLE when no stage is running."""
        assert workflow.status == WorkflowStatus.IDLE

    def test_current_stage_name_none_initially(self, workflow):
        """Test that current_stage_name is None initially."""
        assert workflow.current_stage_name is None

    def test_current_stage_name_after_compose(self, composed_workflow):
        """Test current_stage_name reflects _current_stage."""
        composed_workflow._current_stage = composed_workflow._stage_map["setup"]
        assert composed_workflow.current_stage_name == "setup"


class TestBasicValidation:
    """Tests for BasicDFL graph validation."""

    def test_validate_is_valid(self):
        """Test that the workflow graph is valid."""
        wf = BasicDFL()
        result = validate(wf)
        assert result.is_valid, f"Validation errors: {result.errors}"

    def test_validate_transitions(self):
        """Test that transitions are correctly extracted."""
        wf = BasicDFL()
        result = validate(wf)
        transitions = {k: v.targets for k, v in result.transitions.items()}
        assert "round_init" in transitions["setup"]
        assert "voting" in transitions["round_init"]
        assert "finish" in transitions["round_init"]
        assert "learning_evaluate" in transitions["voting"]
        assert "learning_train" in transitions["learning_evaluate"]
        assert "learning_wait_model" in transitions["learning_evaluate"]
        assert "learning_gossip_loop" in transitions["learning_train"]
        assert "learning_aggregate" in transitions["learning_gossip_loop"]
        assert "round_init" in transitions["learning_aggregate"]
        assert "round_init" in transitions["learning_wait_model"]
        assert transitions["finish"] == {None}


###
# LearningWaitModelStage Tests
###


class TestLearningWaitModelStage:
    """Tests for LearningWaitModelStage run() and handle_add_model()."""

    @pytest.fixture
    def wait_stage(self):
        """Create a LearningWaitModelStage with a mocked context."""
        stage = LearningWaitModelStage()
        ctx = MagicMock(spec=BasicDFLContext)
        ctx.address = "test_node"
        ctx.experiment = Experiment.create(exp_name="test", total_rounds=5, trainset_size=3)
        ctx.experiment.round = 2
        ctx.needs_full_model = True
        ctx.full_model_ready = asyncio.Event()
        ctx.learner = MagicMock()
        stage.ctx = ctx
        return stage

    @pytest.mark.asyncio
    async def test_run_waits_and_advances_round(self, wait_stage):
        """run() waits for full_model_ready, clears needs_full_model, advances round."""
        ctx = wait_stage.ctx
        # Pre-set the event so wait returns immediately
        ctx.full_model_ready.set()

        result = await wait_stage.run()

        assert result == "round_init"
        assert ctx.needs_full_model is False
        assert ctx.experiment.round == 3

    @pytest.mark.asyncio
    async def test_run_timeout_still_advances(self, wait_stage):
        """run() advances round even on timeout."""
        ctx = wait_stage.ctx
        # Patch timeout to a tiny value so it times out fast
        with patch("p2pfl.workflow.basic_dfl.stages.learning_wait_model.Settings") as mock_settings:
            mock_settings.training.AGGREGATION_TIMEOUT = 0.01
            result = await wait_stage.run()

        assert result == "round_init"
        assert ctx.needs_full_model is False
        assert ctx.experiment.round == 3

    @pytest.mark.asyncio
    async def test_handle_add_model_sets_model_and_event(self, wait_stage):
        """handle_add_model sets the model on the learner and fires the event."""
        ctx = wait_stage.ctx
        assert not ctx.full_model_ready.is_set()

        await wait_stage.handle_add_model(
            source="peer1",
            round=2,
            weights=b"model_bytes",
            contributors=["peer1"],
            num_samples=100,
        )

        ctx.learner.set_model.assert_called_once_with(b"model_bytes")
        assert ctx.full_model_ready.is_set()

    @pytest.mark.asyncio
    async def test_handle_add_model_stale_round_ignored(self, wait_stage):
        """handle_add_model ignores messages from a previous round."""
        ctx = wait_stage.ctx

        await wait_stage.handle_add_model(
            source="peer1",
            round=1,  # stale: current round is 2
            weights=b"old_data",
            contributors=["peer1"],
            num_samples=50,
        )

        ctx.learner.set_model.assert_not_called()
        assert not ctx.full_model_ready.is_set()

    @pytest.mark.asyncio
    async def test_handle_add_model_decoding_error(self, wait_stage):
        """handle_add_model catches DecodingParamsError without crashing."""
        ctx = wait_stage.ctx
        ctx.learner.set_model.side_effect = DecodingParamsError("bad bytes")

        await wait_stage.handle_add_model(
            source="peer1",
            round=2,
            weights=b"corrupt",
            contributors=None,
            num_samples=None,
        )

        # Event should NOT be set on error
        assert not ctx.full_model_ready.is_set()

    @pytest.mark.asyncio
    async def test_handle_add_model_model_not_matching(self, wait_stage):
        """handle_add_model catches ModelNotMatchingError without crashing."""
        ctx = wait_stage.ctx
        ctx.learner.set_model.side_effect = ModelNotMatchingError("shape mismatch")

        await wait_stage.handle_add_model(
            source="peer1",
            round=2,
            weights=b"wrong_arch",
            contributors=["peer1"],
            num_samples=100,
        )

        assert not ctx.full_model_ready.is_set()

    @pytest.mark.asyncio
    async def test_handle_add_model_unexpected_error(self, wait_stage):
        """handle_add_model catches generic exceptions without crashing."""
        ctx = wait_stage.ctx
        ctx.learner.set_model.side_effect = RuntimeError("disk full")

        await wait_stage.handle_add_model(
            source="peer1",
            round=2,
            weights=b"data",
            contributors=["peer1"],
            num_samples=100,
        )

        assert not ctx.full_model_ready.is_set()
