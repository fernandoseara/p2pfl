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

"""Tests for workflow base classes."""

import math
from unittest.mock import patch

import pytest

from p2pfl.workflow.engine.experiment import Experiment
from p2pfl.workflow.engine.stage import Stage


class TestExperiment:
    """Tests for Experiment class."""

    def test_create_experiment(self):
        """Test creating an experiment."""
        exp = Experiment(
            exp_name="test_exp",
            total_rounds=10,
            epochs_per_round=2,
            trainset_size=100,
        )

        assert exp.exp_name == "test_exp"
        assert exp.total_rounds == 10
        assert exp.epochs_per_round == 2
        assert exp.trainset_size == 100

    def test_experiment_round_defaults_zero(self):
        """Test that round defaults to 0."""
        exp = Experiment(exp_name="test", total_rounds=5)
        assert exp.round == 0

    def test_experiment_data_defaults_empty(self):
        """Test that data defaults to empty dict."""
        exp = Experiment(exp_name="test", total_rounds=5)
        assert exp.data == {}

    def test_experiment_increase_round(self):
        """Test increase_round increments and calls logger."""
        exp = Experiment(exp_name="test", total_rounds=5)
        with patch("p2pfl.workflow.engine.experiment.logger") as mock_logger:
            exp.increase_round("node1")
            assert exp.round == 1
            mock_logger.round_updated.assert_called_once_with("node1", 1)
            exp.increase_round("node1")
            assert exp.round == 2

    def test_experiment_data_tracking(self):
        """Test that experiment.data can store flexible tracking data."""
        exp = Experiment(exp_name="test", total_rounds=5)
        exp.data["train_history"] = [0.9, 0.85, 0.8]
        exp.data["aggregation_info"] = {"method": "fedavg"}
        assert exp.data["train_history"] == [0.9, 0.85, 0.8]
        assert exp.data["aggregation_info"]["method"] == "fedavg"


class TestExperimentStr:
    """Tests for Experiment.__str__ with edge-case values."""

    def test_str_includes_zero_learning_rate(self):
        """Test that __str__ includes learning_rate even when it is 0.0."""
        exp = Experiment(
            exp_name="test",
            total_rounds=5,
            epochs_per_round=1,
            learning_rate=0.0,
        )
        result = str(exp)
        assert "learning_rate=0.0" in result

    def test_str_includes_zero_batch_size(self):
        """Test that __str__ includes batch_size even when it is 0."""
        exp = Experiment(
            exp_name="test",
            total_rounds=5,
            epochs_per_round=1,
            batch_size=0,
        )
        result = str(exp)
        assert "batch_size=0" in result

    def test_str_excludes_none_learning_rate(self):
        """Test that __str__ excludes learning_rate when it is None."""
        exp = Experiment(
            exp_name="test",
            total_rounds=5,
            epochs_per_round=1,
        )
        result = str(exp)
        assert "learning_rate" not in result


class TestComputePriority:
    """Tests for compute_priority math fixes."""

    def test_dij_clamped_to_one(self):
        """Test that dij is clamped to 1.0 when staleness exceeds dmax."""
        from p2pfl.workflow.async_dfl.stages.training_round import compute_priority

        # staleness = abs((100 - 0) - (0 - 0)) / 5 = 20.0, should be clamped to 1.0
        priority = compute_priority(ti=100, tp_ij=0, tj=0, tl_ji=0, f_ti=0.5, f_tj=0.5, dmax=5)
        # When dij=1.0 and losses are equal, loss_term = exp(0)/exp(1) = 1/e
        # priority = 1.0 + (1-1.0) * loss_term = 1.0
        assert priority == pytest.approx(1.0)

    def test_overflow_protection_on_large_loss_diff(self):
        """Test that math.exp overflow is caught and produces inf."""
        from p2pfl.workflow.async_dfl.stages.training_round import compute_priority

        # Very large loss difference should not crash
        priority = compute_priority(ti=1, tp_ij=0, tj=0, tl_ji=0, f_ti=0.0, f_tj=1000.0, dmax=5)
        assert priority == float("inf") or priority > 1e6  # Should handle gracefully

    def test_dmax_zero_raises_value_error(self):
        """Test that dmax=0 raises ValueError."""
        from p2pfl.workflow.async_dfl.stages.training_round import compute_priority

        with pytest.raises(ValueError, match="dmax must be positive"):
            compute_priority(ti=1, tp_ij=0, tj=0, tl_ji=0, f_ti=0.5, f_tj=0.5, dmax=0)

    def test_perfect_sync_with_equal_losses(self):
        """Test that perfect sync with equal losses gives the base priority."""
        from p2pfl.workflow.async_dfl.stages.training_round import compute_priority

        # dij = 0, loss_term = exp(0)/exp(1) = 1/e
        # priority = 0 + (1-0) * 1/e = 1/e
        priority = compute_priority(ti=5, tp_ij=5, tj=5, tl_ji=5, f_ti=0.5, f_tj=0.5, dmax=5)
        assert priority == pytest.approx(1.0 / math.e)


class TestWorkflowExperimentAndRound:
    """Tests for Workflow experiment/round management."""

    def test_workflow_defaults(self):
        """Test that workflow defaults are correct."""
        from p2pfl.workflow.basic_dfl.workflow import BasicDFL

        wf = BasicDFL()
        assert wf.experiment is None

    def test_experiment_properties(self):
        """Test experiment property accessors on workflow."""
        from p2pfl.workflow.basic_dfl.workflow import BasicDFL

        wf = BasicDFL()
        wf.experiment = Experiment("test", 5, epochs_per_round=2)
        assert wf.experiment.exp_name == "test"
        assert wf.experiment.total_rounds == 5
        assert wf.experiment.epochs_per_round == 2
        assert wf.experiment.round == 0

    def test_increase_round(self):
        """Test round increment via experiment."""
        exp = Experiment("test", 5, epochs_per_round=1)
        with patch("p2pfl.workflow.engine.experiment.logger"):
            exp.increase_round("node1")
            assert exp.round == 1
            exp.increase_round("node1")
            assert exp.round == 2


class TestStage:
    """Tests for Stage base class."""

    def test_stage_is_abstract(self):
        """Test that Stage cannot be instantiated directly (run is abstract)."""
        with pytest.raises(TypeError):
            Stage()  # type: ignore[abstract]

    def test_stage_has_run_method(self):
        """Test that Stage defines abstract run method."""
        assert hasattr(Stage, "run")
