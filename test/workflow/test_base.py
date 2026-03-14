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

from unittest.mock import MagicMock

import pytest

from p2pfl.workflow.engine.experiment import Experiment
from p2pfl.workflow.engine.observable import Observer
from p2pfl.workflow.engine.stage import Stage


class TestExperiment:
    """Tests for Experiment class."""

    def test_create_experiment(self):
        """Test creating an experiment with extra kwargs as dynamic attributes."""
        exp = Experiment.create(
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

    def test_experiment_round_fires_observer(self):
        """Test that setting round notifies observers."""
        exp = Experiment(exp_name="test", total_rounds=5)
        observer = MagicMock(spec=Observer)
        exp.add_observer(observer)

        exp.round = 1
        observer.update.assert_called_with("round", 1)

        exp.round = 2
        assert observer.update.call_count == 2
        observer.update.assert_called_with("round", 2)

    def test_experiment_dynamic_attrs_fire_observer(self):
        """Test that setting dynamic attributes notifies observers."""
        exp = Experiment.create(exp_name="test", total_rounds=5, tau=3)
        observer = MagicMock(spec=Observer)
        exp.add_observer(observer)

        exp.tau = 5
        observer.update.assert_called_with("tau", 5)

    def test_experiment_dynamic_attrs_in_to_dict(self):
        """Test that dynamic attributes appear in to_dict."""
        exp = Experiment.create(exp_name="test", total_rounds=5, trainset_size=100, tau=3)
        d = exp.to_dict()
        assert d["trainset_size"] == 100
        assert d["tau"] == 3

    def test_experiment_to_dict_excludes_none(self):
        """Test that to_dict excludes None values by default."""
        exp = Experiment(exp_name="test", total_rounds=5)
        d = exp.to_dict()
        assert "dataset_name" not in d

    def test_experiment_to_dict_includes_none(self):
        """Test that to_dict includes None values when exclude_none=False."""
        exp = Experiment(exp_name="test", total_rounds=5)
        d = exp.to_dict(exclude_none=False)
        assert "dataset_name" in d
        assert d["dataset_name"] is None


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


class TestObservable:
    """Tests for Observable mixin."""

    def test_add_and_notify(self):
        """Test that observers are notified on setattr."""
        exp = Experiment(exp_name="test", total_rounds=5)
        observer = MagicMock(spec=Observer)
        exp.add_observer(observer)
        exp.round = 3
        observer.update.assert_called_with("round", 3)

    def test_remove_observer(self):
        """Test that removed observers stop receiving notifications."""
        exp = Experiment(exp_name="test", total_rounds=5)
        observer = MagicMock(spec=Observer)
        exp.add_observer(observer)
        exp.remove_observer(observer)
        exp.round = 3
        observer.update.assert_not_called()

    def test_clear_observers(self):
        """Test that clear_observers removes all observers."""
        exp = Experiment(exp_name="test", total_rounds=5)
        obs1 = MagicMock(spec=Observer)
        obs2 = MagicMock(spec=Observer)
        exp.add_observer(obs1)
        exp.add_observer(obs2)
        exp.clear_observers()
        exp.round = 3
        obs1.update.assert_not_called()
        obs2.update.assert_not_called()

    def test_underscore_attrs_not_notified(self):
        """Test that _-prefixed attributes don't trigger observers."""
        exp = Experiment(exp_name="test", total_rounds=5)
        observer = MagicMock(spec=Observer)
        exp.add_observer(observer)
        exp._internal = "secret"
        observer.update.assert_not_called()

    def test_no_notifications_during_init(self):
        """Test that dataclass __init__ doesn't trigger observer notifications."""
        observer = MagicMock(spec=Observer)
        exp = Experiment(exp_name="test", total_rounds=5)
        # Observer added after init — no prior notifications
        exp.add_observer(observer)
        observer.update.assert_not_called()


class TestStage:
    """Tests for Stage base class."""

    def test_stage_is_abstract(self):
        """Test that Stage cannot be instantiated directly (run is abstract)."""
        with pytest.raises(TypeError):
            Stage()  # type: ignore[abstract]
