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

from unittest.mock import MagicMock, patch

import pytest

from p2pfl.workflow.basic_dfl.context import BasicDFLContext, BasicPeerState
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
        expected = {"setup", "round_init", "voting", "learning", "finish"}
        assert {s.name for s in stages} == expected

    def test_stage_map_types(self):
        """Test that each stage is the correct type."""
        from p2pfl.workflow.basic_dfl.stages import (
            FinishStage,
            LearningStage,
            RoundInitStage,
            SetupStage,
            VotingStage,
        )

        wf = BasicDFL()
        stages = {s.name: s for s in wf.get_stages()}

        assert isinstance(stages["setup"], SetupStage)
        assert isinstance(stages["round_init"], RoundInitStage)
        assert isinstance(stages["voting"], VotingStage)
        assert isinstance(stages["learning"], LearningStage)
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
        assert msgs["add_model"].during is None
        assert msgs["models_aggregated"].during is None
        assert msgs["pre_send_model_init"].during == frozenset({"round_init"})
        assert msgs["pre_send_model_learning"].during is None
        assert msgs["partial_model"].during is None
        # Handlers with explicit during=
        assert msgs["peer_round_updated"].during == frozenset({"round_init", "learning", "voting"})
        assert msgs["vote_train_set"].during == frozenset({"voting", "round_init"})


class TestBasicMessageRegistry:
    """Tests for message registry (after composition)."""

    def test_registry_contains_all_messages(self, composed_workflow):
        """Test that the message registry contains all expected messages."""
        registry = composed_workflow.get_messages()
        expected_messages = {
            "node_initialized",
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


class TestBasicConditions:
    """Tests for workflow condition methods."""

    def test_all_nodes_started(self, composed_workflow, ctx):
        """Test _all_nodes_started condition via setup stage."""
        ctx.peers["node_1"] = BasicPeerState()
        ctx.peers["node_2"] = BasicPeerState()
        ctx.peers["node_3"] = BasicPeerState()

        ctx.cp.get_neighbors.return_value = {"node_2": {}, "node_3": {}}

        setup_stage = composed_workflow._stage_map["setup"]
        assert setup_stage._all_nodes_started(ctx)

    def test_all_nodes_started_false_when_missing(self, composed_workflow, ctx):
        """Test _all_nodes_started returns false when peers are missing."""
        ctx.peers["node_1"] = BasicPeerState()
        ctx.cp.get_neighbors.return_value = {"node_2": {}, "node_3": {}}

        setup_stage = composed_workflow._stage_map["setup"]
        assert not setup_stage._all_nodes_started(ctx)

    def test_in_train_set(self, composed_workflow, ctx):
        """Test _in_train_set condition via voting stage."""
        voting_stage = composed_workflow._stage_map["voting"]

        ctx.train_set = ["node_1", "node_2"]
        ctx.address = "node_1"
        assert voting_stage._in_train_set(ctx)

        ctx.address = "node_3"
        assert not voting_stage._in_train_set(ctx)

    def test_total_rounds_reached_false_when_total_rounds_is_none(self, composed_workflow, ctx):
        """Test that _total_rounds_reached returns False when total_rounds is None."""
        ctx.experiment.total_rounds = None
        round_init_stage = composed_workflow._stage_map["round_init"]
        assert round_init_stage._total_rounds_reached(ctx) is False

    def test_total_rounds_reached_true_when_reached(self, composed_workflow, ctx):
        """Test that _total_rounds_reached returns True when round >= total_rounds."""
        ctx.experiment = Experiment("test", 2, epochs_per_round=1)
        with patch("p2pfl.workflow.engine.experiment.logger"):
            ctx.experiment.increase_round("node1")
            ctx.experiment.increase_round("node1")
            round_init_stage = composed_workflow._stage_map["round_init"]
            assert round_init_stage._total_rounds_reached(ctx) is True

    def test_all_votes_received(self, composed_workflow, ctx):
        """Test _all_votes_received condition via voting stage."""
        voting_stage = composed_workflow._stage_map["voting"]

        ctx.peers["node_1"] = BasicPeerState()
        ctx.peers["node_2"] = BasicPeerState()
        assert not voting_stage._all_votes_received(ctx)

        ctx.peers["node_1"].votes = {"a": 1}
        assert not voting_stage._all_votes_received(ctx)

        ctx.peers["node_2"].votes = {"b": 2}
        assert voting_stage._all_votes_received(ctx)

    def test_all_models_received(self, composed_workflow, ctx):
        """Test _all_models_received condition via learning stage."""
        learning_stage = composed_workflow._stage_map["learning"]
        ctx.train_set = ["node_1", "node_2"]

        ctx.peers["node_1"] = BasicPeerState()
        ctx.peers["node_2"] = BasicPeerState()
        assert not learning_stage._all_models_received(ctx)

        ctx.peers["node_1"].model = MagicMock()
        ctx.peers["node_2"].model = MagicMock()
        assert learning_stage._all_models_received(ctx)


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
        assert "learning" in transitions["voting"]
        assert "round_init" in transitions["voting"]
        assert "round_init" in transitions["learning"]
        assert transitions["finish"] == {None}
