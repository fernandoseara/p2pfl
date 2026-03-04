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

"""Tests for AsyncDFL workflow (new engine)."""

from unittest.mock import MagicMock, patch

import pytest

from p2pfl.workflow.async_dfl.context import AsyncDFLContext, AsyncPeerState
from p2pfl.workflow.async_dfl.stages.training_round import TrainingRoundStage, compute_priority
from p2pfl.workflow.async_dfl.workflow import AsyncDFL
from p2pfl.workflow.engine.experiment import Experiment
from p2pfl.workflow.factory import create_workflow
from p2pfl.workflow.validation import validate


class TestAsyncDFLCreation:
    """Tests for AsyncDFL workflow creation and initialization."""

    def test_init_returns_workflow(self):
        """Test that AsyncDFL can be instantiated."""
        wf = AsyncDFL()
        assert isinstance(wf, AsyncDFL)

    def test_initial_stage_is_setup(self):
        """Test that initial_stage is 'setup'."""
        wf = AsyncDFL()
        assert wf.initial_stage == "setup"

    def test_stages_map_has_all_stages(self):
        """Test that get_stages returns all expected stages."""
        wf = AsyncDFL()
        stages = wf.get_stages()
        assert {s.name for s in stages} == {"setup", "training_round", "finish"}

    def test_factory_creates_async(self):
        """Test that factory creates AsyncDFL for ASYNC type."""
        wf = create_workflow("async")
        assert isinstance(wf, AsyncDFL)


class TestAsyncDFLValidation:
    """Tests for AsyncDFL graph validation."""

    def test_validate_is_valid(self):
        """Test that the workflow graph is valid."""
        wf = AsyncDFL()
        result = validate(wf)
        assert result.is_valid, f"Validation errors: {result.errors}"

    def test_validate_transitions(self):
        """Test that transitions are correctly extracted."""
        wf = AsyncDFL()
        result = validate(wf)
        transitions = {k: v.targets for k, v in result.transitions.items()}
        assert "training_round" in transitions["setup"]
        assert "training_round" in transitions["training_round"]
        assert "finish" in transitions["training_round"]
        assert transitions["finish"] == {None}


class TestAsyncDFLDeclaredMessages:
    """Tests for message declaration (before run)."""

    def test_declared_messages_contain_all(self):
        """Test that get_messages returns all expected messages."""
        wf = AsyncDFL()
        msgs = wf.get_messages()
        expected = {
            "node_initialized",
            "loss_information_updating",
            "index_information_updating",
            "model_information_updating",
            "push_sum_weight_information_updating",
            "pre_send_model",
        }
        assert set(msgs.keys()) == expected

    def test_weights_messages_flagged(self):
        """Test that weight messages are properly flagged."""
        wf = AsyncDFL()
        msgs = wf.get_messages()
        assert msgs["model_information_updating"].is_weights is True
        assert msgs["node_initialized"].is_weights is False
        assert msgs["loss_information_updating"].is_weights is False
        assert msgs["push_sum_weight_information_updating"].is_weights is False

    def test_during_filters_set(self):
        """Test that during filters are set on all handlers."""
        wf = AsyncDFL()
        msgs = wf.get_messages()
        assert msgs["node_initialized"].during == frozenset({"setup"})
        assert msgs["loss_information_updating"].during == frozenset({"training_round"})
        assert msgs["model_information_updating"].during == frozenset({"training_round"})
        assert msgs["pre_send_model"].during == frozenset({"training_round"})


class TestAsyncPeerState:
    """Tests for AsyncPeerState operations."""

    def test_default_values(self):
        """Test that AsyncPeerState has correct defaults."""
        peer = AsyncPeerState()
        assert peer.round_number == 0
        assert peer.push_sum_weight == 1.0
        assert peer.model is None
        assert peer.losses == {}
        assert peer.push_time == 0
        assert peer.mixing_weight == 1.0
        assert peer.p2p_updating_idx == 0

    def test_add_loss(self):
        """Test add_loss sets loss at round index in dict."""
        peer = AsyncPeerState()
        peer.add_loss(0, 0.5)
        assert peer.losses == {0: 0.5}
        peer.add_loss(3, 0.2)
        assert peer.losses == {0: 0.5, 3: 0.2}

    def test_add_loss_overwrites(self):
        """Test add_loss overwrites existing value."""
        peer = AsyncPeerState()
        peer.add_loss(0, 0.5)
        peer.add_loss(0, 0.9)
        assert peer.losses == {0: 0.9}

    def test_reset_round(self):
        """Test reset_round clears model."""
        peer = AsyncPeerState()
        peer.model = MagicMock()
        peer.reset_round()
        assert peer.model is None

    def test_peers_add_and_clear(self):
        """Test adding and clearing peers."""
        peers: dict[str, AsyncPeerState] = {}
        peers["node_1"] = AsyncPeerState()
        peers["node_2"] = AsyncPeerState()
        assert len(peers) == 2
        peers.clear()
        assert len(peers) == 0


class TestComputePriority:
    """Tests for the compute_priority function."""

    def test_basic_priority(self):
        """Test basic priority computation."""
        p = compute_priority(ti=10, tp_ij=5, tj=8, tl_ji=3, f_ti=0.5, f_tj=0.5, dmax=5)
        assert isinstance(p, float)
        assert p >= 0

    def test_zero_loss_difference(self):
        """Test priority with zero loss difference."""
        p = compute_priority(ti=0, tp_ij=0, tj=0, tl_ji=0, f_ti=1.0, f_tj=1.0, dmax=5)
        # dij = 0, loss_term = exp(0)/exp(1) ≈ 0.368
        assert abs(p - 0.368) < 0.01

    def test_high_staleness(self):
        """Test priority with high staleness."""
        p = compute_priority(ti=10, tp_ij=0, tj=0, tl_ji=0, f_ti=0.5, f_tj=0.5, dmax=5)
        # dij = min(10/5, 1.0) = 1.0, so priority = 1.0
        assert abs(p - 1.0) < 0.01

    def test_dmax_must_be_positive(self):
        """Test that dmax <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="dmax must be positive"):
            compute_priority(ti=0, tp_ij=0, tj=0, tl_ji=0, f_ti=0.0, f_tj=0.0, dmax=0)

    def test_overflow_handled(self):
        """Test that large loss differences don't crash."""
        p = compute_priority(ti=0, tp_ij=0, tj=0, tl_ji=0, f_ti=0.0, f_tj=1000.0, dmax=5)
        assert p == float("inf")


class TestTrainingRoundStageConditions:
    """Tests for TrainingRoundStage condition helpers."""

    def test_total_rounds_reached_false(self):
        """Test _total_rounds_reached returns False when not reached."""
        stage = TrainingRoundStage()
        ctx = MagicMock(spec=AsyncDFLContext)
        ctx.experiment = Experiment("test", total_rounds=5)
        ctx.experiment.round = 2
        assert stage._total_rounds_reached(ctx) is False

    def test_total_rounds_reached_true(self):
        """Test _total_rounds_reached returns True when reached."""
        stage = TrainingRoundStage()
        ctx = MagicMock(spec=AsyncDFLContext)
        ctx.experiment = Experiment("test", total_rounds=5)
        with patch("p2pfl.workflow.engine.experiment.logger"):
            for _ in range(5):
                ctx.experiment.increase_round("test")
        assert stage._total_rounds_reached(ctx) is True

    def test_total_rounds_reached_false_when_none(self):
        """Test _total_rounds_reached returns False when total_rounds is None."""
        stage = TrainingRoundStage()
        ctx = MagicMock(spec=AsyncDFLContext)
        ctx.experiment = MagicMock()
        ctx.experiment.total_rounds = None
        assert stage._total_rounds_reached(ctx) is False

    def test_select_neighbors_top_3(self):
        """Test _select_neighbors picks top 3 by priority."""
        priorities = [("a", 1.0), ("b", 3.0), ("c", 2.0), ("d", 0.5), ("e", 4.0)]
        result = TrainingRoundStage._select_neighbors(priorities)
        assert result == ["e", "b", "c"]

    def test_select_neighbors_fewer_than_3(self):
        """Test _select_neighbors with fewer than 3 neighbors."""
        priorities = [("a", 1.0), ("b", 2.0)]
        result = TrainingRoundStage._select_neighbors(priorities)
        assert result == ["b", "a"]


class TestAsyncDFLContext:
    """Tests for AsyncDFLContext creation."""

    def test_create_context(self):
        """Test that create_context builds a proper AsyncDFLContext."""
        wf = AsyncDFL()
        ctx = wf.create_context(
            address="test",
            learner=MagicMock(),
            aggregator=MagicMock(),
            cp=MagicMock(),
            generator=MagicMock(),
            experiment=Experiment("test", total_rounds=5),
            tau=3,
        )
        assert isinstance(ctx, AsyncDFLContext)
        assert ctx.address == "test"
        assert ctx.tau == 3
        assert ctx.peers == {}
        assert ctx.candidates == []

    def test_create_context_default_tau(self):
        """Test that create_context defaults tau to 2."""
        wf = AsyncDFL()
        ctx = wf.create_context(
            address="test",
            learner=MagicMock(),
            aggregator=MagicMock(),
            cp=MagicMock(),
            generator=MagicMock(),
            experiment=Experiment("test", total_rounds=5),
        )
        assert ctx.tau == 2
