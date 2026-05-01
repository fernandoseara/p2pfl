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

import asyncio
import random
from unittest.mock import MagicMock

import pytest

from p2pfl.management.logger import logger
from p2pfl.workflow.engine.context import WorkflowContext
from p2pfl.workflow.engine.experiment import Experiment
from p2pfl.workflow.engine.message import on_message
from p2pfl.workflow.engine.observable import Observer
from p2pfl.workflow.engine.stage import Stage
from p2pfl.workflow.engine.workflow import Workflow, WorkflowStatus


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


###
# Helpers for Workflow tests
###


class _StageA(Stage[WorkflowContext]):
    name = "stage_a"

    async def run(self) -> str | None:
        return "stage_b"


class _StageB(Stage[WorkflowContext]):
    name = "stage_b"

    async def run(self) -> str | None:
        return None


class _FailingStage(Stage[WorkflowContext]):
    name = "failing"

    async def run(self) -> str | None:
        raise RuntimeError("stage exploded")
        return None  # noqa: RET504  # unreachable, needed for AST validation


class _StageWithWeightsHandler(Stage[WorkflowContext]):
    name = "weights_ok"

    @on_message("model_weights", weights=True)
    async def handle_weights(self, source, round, weights, contributors, num_samples):
        pass

    async def run(self) -> str | None:
        return None


class _StageWithBadWeightsHandler(Stage[WorkflowContext]):
    name = "bad_weights_stage"

    @on_message("bad_weights", weights=True)
    async def handle_bad(self, source, round):
        pass

    async def run(self) -> str | None:
        return None


class _StageWithBadDuring(Stage[WorkflowContext]):
    name = "bad_during"

    @on_message("bad_during_msg", during={"nonexistent_stage"})
    async def handle(self, source, round, *args):
        pass

    async def run(self) -> str | None:
        return None


def _make_workflow(stages):
    """Create a concrete Workflow subclass with given stages."""

    class _TestWorkflow(Workflow[WorkflowContext]):
        context_class = WorkflowContext

        def get_stages(self):
            return stages

    return _TestWorkflow()


###
# WorkflowStatus tests
###


class TestWorkflowStatus:
    """Tests for WorkflowStatus.is_terminal property."""

    def test_is_terminal_for_terminal_states(self):
        """Test is terminal for terminal states."""
        assert WorkflowStatus.FINISHED.is_terminal is True
        assert WorkflowStatus.CANCELLED.is_terminal is True
        assert WorkflowStatus.FAILED.is_terminal is True

    def test_is_terminal_for_non_terminal_states(self):
        """Test is terminal for non terminal states."""
        assert WorkflowStatus.IDLE.is_terminal is False
        assert WorkflowStatus.RUNNING.is_terminal is False


###
# Workflow composition and validation tests
###


class TestWorkflowComposition:
    """Tests for workflow composition, handler registration, and graph validation."""

    def _make_ctx(self):
        return WorkflowContext(
            address="test_node",
            learner=MagicMock(),
            aggregator=MagicMock(),
            cp=MagicMock(),
            generator=random.Random(42),
            experiment=Experiment(exp_name="test", total_rounds=1),
        )

    def test_initial_stage_empty_raises(self):
        """get_stages() returning [] raises ValueError from initial_stage."""
        wf = _make_workflow([])
        with pytest.raises(ValueError, match="empty list"):
            _ = wf.initial_stage

    def test_invalid_graph_raises(self):
        """Stage returning a nonexistent target triggers validation error."""

        class _BadStage(Stage[WorkflowContext]):
            async def run(self) -> str | None:
                return "does_not_exist"

        wf = _make_workflow([_BadStage()])
        with pytest.raises(ValueError, match="Invalid workflow graph"):
            wf._compose(self._make_ctx())

    def test_weights_handler_missing_param_raises(self):
        """weights=True handler without 'weights' param raises at compose time."""
        wf = _make_workflow([_StageWithBadWeightsHandler()])
        with pytest.raises(ValueError, match="lacks a 'weights' parameter"):
            wf._compose(self._make_ctx())

    def test_handler_collision_raises(self):
        """Two handlers for same message with overlapping during sets raise."""

        class _StageX(Stage[WorkflowContext]):
            name = "stage_x"

            @on_message("collision_msg")
            async def handle(self, source, round, *args):
                pass

            async def run(self) -> str | None:
                return "stage_y"

        class _StageY(Stage[WorkflowContext]):
            name = "stage_y"

            @on_message("collision_msg", during={"stage_x"})
            async def handle(self, source, round, *args):
                pass

            async def run(self) -> str | None:
                return None

        wf = _make_workflow([_StageX(), _StageY()])
        with pytest.raises(ValueError, match="Handler collision"):
            wf._compose(self._make_ctx())

    def test_bad_during_name_raises(self):
        """During referencing nonexistent stage raises with available-stage listing."""
        wf = _make_workflow([_StageA(), _StageB(), _StageWithBadDuring()])
        with pytest.raises(ValueError, match="does not match any stage"):
            wf._compose(self._make_ctx())

    def test_expand_during_regex(self):
        """_expand_during resolves regex patterns against registered stage names."""
        wf = _make_workflow([_StageA(), _StageB()])
        wf._compose(self._make_ctx())
        expanded = wf._expand_during(frozenset({"stage_.*"}))
        assert "stage_a" in expanded
        assert "stage_b" in expanded

    def test_expand_during_none_returns_all(self):
        """_expand_during(None) returns all stage names."""
        wf = _make_workflow([_StageA(), _StageB()])
        wf._compose(self._make_ctx())
        expanded = wf._expand_during(None)
        assert expanded == set(wf._stage_map.keys())


###
# Workflow run tests
###

_TEST_NODE = "wf_test_node"


def _run_kwargs(experiment=None):
    """Build common kwargs for Workflow.run()."""
    if experiment is None:
        experiment = Experiment(exp_name="test", total_rounds=1)
    return {
        "experiment": experiment,
        "address": _TEST_NODE,
        "learner": MagicMock(),
        "aggregator": MagicMock(),
        "cp": MagicMock(),
        "generator": random.Random(42),
    }


@pytest.fixture(autouse=True, scope="module")
def _register_test_node():
    """Register and unregister a test node in the logger for workflow tests."""
    logger.register_node(_TEST_NODE)
    yield
    logger.unregister_node(_TEST_NODE)


class TestWorkflowRun:
    """Tests for workflow run lifecycle: success, failure, and cancellation."""

    @pytest.mark.asyncio
    async def test_run_sets_finished(self):
        """Test run sets finished."""
        wf = _make_workflow([_StageA(), _StageB()])
        experiment = Experiment(exp_name="test", total_rounds=1)
        result = await wf.run(**_run_kwargs(experiment))
        assert wf.status == WorkflowStatus.FINISHED
        assert result is experiment

    @pytest.mark.asyncio
    async def test_run_failure_sets_failed_and_stores_error(self):
        """Test run failure sets failed and stores error."""
        wf = _make_workflow([_FailingStage()])
        with pytest.raises(RuntimeError, match="stage exploded"):
            await wf.run(**_run_kwargs())
        assert wf.status == WorkflowStatus.FAILED
        assert isinstance(wf.error, RuntimeError)

    @pytest.mark.asyncio
    async def test_run_cancellation_sets_cancelled(self):
        """Test run cancellation sets cancelled."""

        class _SlowStage(Stage[WorkflowContext]):
            name = "slow"

            async def run(self) -> str | None:
                await asyncio.sleep(100)
                return None

        wf = _make_workflow([_SlowStage()])
        task = asyncio.create_task(wf.run(**_run_kwargs()))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert wf.status == WorkflowStatus.CANCELLED


###
# Task management tests
###


class TestWorkflowTaskManagement:
    """Tests for start/stop/wait task management."""

    @pytest.mark.asyncio
    async def test_start_already_running_raises(self):
        """Test start already running raises."""

        class _SlowStage(Stage[WorkflowContext]):
            name = "slow"

            async def run(self) -> str | None:
                await asyncio.sleep(100)
                return None

        wf = _make_workflow([_SlowStage()])
        await wf.start(**_run_kwargs())
        try:
            with pytest.raises(RuntimeError, match="already running"):
                await wf.start(**_run_kwargs())
        finally:
            await wf.stop()

    @pytest.mark.asyncio
    async def test_wait_not_started_raises(self):
        """Test wait not started raises."""
        wf = _make_workflow([_StageA(), _StageB()])
        with pytest.raises(RuntimeError, match="not started"):
            await wf.wait()

    @pytest.mark.asyncio
    async def test_stop_done_task_with_exception(self):
        """Stopping a done-but-failed task retrieves the exception silently."""
        wf = _make_workflow([_FailingStage()])
        await wf.start(**_run_kwargs())
        await asyncio.sleep(0.15)
        # stop() on a done+failed task should not raise
        await wf.stop()
        assert wf._task is None


###
# Properties tests
###


class TestWorkflowProperties:
    """Tests for workflow property accessors."""

    def test_experiment_property_idle(self):
        """Test experiment property idle."""
        wf = _make_workflow([_StageA(), _StageB()])
        assert wf.experiment is None

    def test_current_stage_name_idle(self):
        """Test current stage name idle."""
        wf = _make_workflow([_StageA(), _StageB()])
        assert wf.current_stage_name is None

    def test_get_messages_pre_compose(self):
        """get_messages works before _compose from class-level registries."""
        wf = _make_workflow([_StageWithWeightsHandler()])
        msgs = wf.get_messages()
        assert "model_weights" in msgs
        assert msgs["model_weights"].is_weights is True
