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

from unittest.mock import patch

import pytest

from p2pfl.workflow.engine.experiment import Experiment
from p2pfl.workflow.engine.stage import Stage


class TestExperiment:
    """Tests for Experiment class."""

    def test_create_experiment(self):
        """Test creating an experiment."""
        exp = Experiment.create(
            exp_name="test_exp",
            total_rounds=10,
            epochs_per_round=2,
            trainset_size=100,
        )

        assert exp.exp_name == "test_exp"
        assert exp.total_rounds == 10
        assert exp.epochs_per_round == 2
        assert exp.data["trainset_size"] == 100

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


class TestStage:
    """Tests for Stage base class."""

    def test_stage_is_abstract(self):
        """Test that Stage cannot be instantiated directly (run is abstract)."""
        with pytest.raises(TypeError):
            Stage()  # type: ignore[abstract]
