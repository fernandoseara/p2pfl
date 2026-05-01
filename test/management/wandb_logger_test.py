"""Tests for WandbLogger decorator."""

from __future__ import annotations

import logging
import types
from typing import Any
from unittest.mock import MagicMock, patch

import p2pfl.management.logger.decorators.wandb_logger as wl_mod
from p2pfl.management.logger.decorators.wandb_logger import WandbLogger
from p2pfl.workflow.engine.experiment import Experiment

###
# Helpers
###


def _make_mock_wandb() -> types.ModuleType:
    """Build a fake ``wandb`` module with the attributes WandbLogger uses."""
    mod = types.ModuleType("wandb")
    mod.init = MagicMock(name="wandb.init")  # type: ignore[attr-defined]
    mod.log = MagicMock(name="wandb.log")  # type: ignore[attr-defined]
    mod.finish = MagicMock(name="wandb.finish")  # type: ignore[attr-defined]
    mod.config = MagicMock(name="wandb.config")  # type: ignore[attr-defined]
    return mod


def _make_mock_run() -> MagicMock:
    return MagicMock(name="wandb_run")


def _make_experiment(**overrides: Any) -> Experiment:
    kw: dict[str, Any] = {"exp_name": "test_exp", "total_rounds": 3}
    kw.update(overrides)
    return Experiment(**kw)


def _make_base_mock() -> MagicMock:
    """
    Create a mock standing in for P2PFLogger.

    LoggerDecorator.__init__ calls ``logger() if callable(logger) else logger``.
    Since MagicMock is callable, we make ``base()`` return ``base`` itself so
    that ``self._p2pfl_logger`` is the same object we hold a reference to.
    """
    base = MagicMock()
    base.return_value = base
    return base


def _base_logged_at(base: MagicMock, level: int) -> bool:
    """
    Check whether the base mock received a ``log(level, ...)`` call.

    ``super().warning(node, msg)`` in WandbLogger chains through
    ``P2PFLogger.warning`` -> ``self.log(WARNING, ...)`` ->
    ``LoggerDecorator.log`` -> ``self._p2pfl_logger.log(WARNING, ...)``.
    So the base mock records the call on ``.log``, not ``.warning``.
    """
    return any(c for c in base.log.call_args_list if c[0][0] == level)


def _build_logger(
    mock_wandb: types.ModuleType | None = None,
    env: dict[str, str] | None = None,
    connect_kwargs: dict | None = None,
):
    """
    Create a WandbLogger with module-level ``wandb_module`` replaced.

    Returns (logger, base_mock, mock_wandb).
    """
    if mock_wandb is None:
        mock_wandb = _make_mock_wandb()

    base = _make_base_mock()
    env_vars = env or {}

    with (
        patch.dict("os.environ", env_vars, clear=True),
        patch.object(wl_mod, "wandb_module", mock_wandb),
        patch.object(wl_mod, "RunClass", MagicMock),
    ):
        logger = WandbLogger(base)
        if connect_kwargs is not None:
            logger.connect(**connect_kwargs)

    return logger, base, mock_wandb


###
# Tests: connect()
###


class TestConnect:
    """Tests for connect() and the auto-connect in __init__."""

    def test_stores_credentials_from_env(self):
        """Test stores credentials from env."""
        logger, _, _ = _build_logger(env={"WANDB_API_KEY": "test-key-123"})
        assert logger._connected is True
        assert logger._api_key == "test-key-123"
        assert logger._project == "p2pfl"

    def test_stores_all_explicit_kwargs(self):
        """Test stores all explicit kwargs."""
        logger, _, _ = _build_logger(
            env={"WANDB_API_KEY": "k"},
            connect_kwargs={
                "wandb_api_key": "explicit",
                "wandb_project": "proj",
                "wandb_entity": "team",
                "wandb_run_name": "run-1",
                "wandb_tags": ["a", "b"],
                "wandb_notes": "notes",
                "wandb_group": "grp",
            },
        )
        assert logger._api_key == "explicit"
        assert logger._project == "proj"
        assert logger._entity == "team"
        assert logger._run_name == "run-1"
        assert logger._tags == ["a", "b"]
        assert logger._notes == "notes"
        assert logger._group == "grp"

    def test_parses_comma_separated_tags_string(self):
        """Test parses comma separated tags string."""
        logger, _, _ = _build_logger(
            env={"WANDB_API_KEY": "k"},
            connect_kwargs={"wandb_tags": "tag1, tag2, tag3"},
        )
        assert logger._tags == ["tag1", "tag2", "tag3"]

    def test_tags_list_stored_directly(self):
        """Test tags list stored directly."""
        logger, _, _ = _build_logger(
            env={"WANDB_API_KEY": "k"},
            connect_kwargs={"wandb_tags": ["x", "y"]},
        )
        assert logger._tags == ["x", "y"]

    def test_warns_when_project_given_but_no_api_key(self):
        """Test warns when project given but no api key."""
        logger, base, _ = _build_logger(
            env={},
            connect_kwargs={"wandb_project": "proj", "wandb_entity": "ent"},
        )
        assert logger._connected is False
        assert _base_logged_at(base, logging.WARNING)

    def test_silent_when_no_api_key_and_no_project(self):
        """Test silent when no api key and no project."""
        logger, base, _ = _build_logger(env={})
        assert logger._connected is False
        assert not _base_logged_at(base, logging.WARNING)

    def test_disabled_when_wandb_not_installed(self):
        """Test disabled when wandb not installed."""
        base = _make_base_mock()
        with patch.dict("os.environ", {}, clear=True), patch.object(wl_mod, "wandb_module", None), patch.object(wl_mod, "RunClass", None):
            logger = WandbLogger(base)
        assert logger._wandb_enabled is False
        assert logger._connected is False

    def test_warns_when_wandb_not_installed_but_project_given(self):
        """Test warns when wandb not installed but project given."""
        base = _make_base_mock()
        with patch.dict("os.environ", {}, clear=True), patch.object(wl_mod, "wandb_module", None), patch.object(wl_mod, "RunClass", None):
            logger = WandbLogger(base)
            logger.connect(wandb_project="should-warn")
        assert _base_logged_at(base, logging.DEBUG)

    def test_reads_all_env_var_fallbacks(self):
        """Test reads all env var fallbacks."""
        env = {
            "WANDB_API_KEY": "k",
            "WANDB_PROJECT": "env-proj",
            "WANDB_ENTITY": "env-ent",
            "WANDB_RUN_GROUP": "env-grp",
            "WANDB_NOTES": "env-notes",
            "WANDB_RUN_NAME": "env-run",
            "WANDB_TAGS": "x,y",
        }
        logger, _, _ = _build_logger(env=env)
        assert logger._project == "env-proj"
        assert logger._entity == "env-ent"
        assert logger._group == "env-grp"
        assert logger._notes == "env-notes"
        assert logger._run_name == "env-run"
        assert logger._tags == ["x", "y"]

    def test_connected_flag_and_debug_message(self):
        """Successful connect sets _connected and emits a debug log."""
        logger, base, _ = _build_logger(env={"WANDB_API_KEY": "k"})
        assert logger._connected is True
        assert _base_logged_at(base, logging.DEBUG)


###
# Tests: experiment_started()
###


class TestExperimentStarted:
    """Experiment Started tests."""

    def test_initializes_wandb_run(self):
        """Test initializes wandb run."""
        mock_wandb = _make_mock_wandb()
        mock_run = _make_mock_run()
        mock_wandb.init.return_value = mock_run  # type: ignore[attr-defined]

        logger, base, _ = _build_logger(
            mock_wandb=mock_wandb,
            env={"WANDB_API_KEY": "k"},
            connect_kwargs={
                "wandb_project": "proj",
                "wandb_entity": "ent",
                "wandb_tags": ["t1"],
                "wandb_notes": "n",
                "wandb_group": "g",
            },
        )
        exp = _make_experiment(exp_name="my_exp")

        with patch.object(wl_mod, "wandb_module", mock_wandb):
            logger.experiment_started("node1", exp)

        mock_wandb.init.assert_called_once()
        kw = mock_wandb.init.call_args[1]
        assert kw["project"] == "proj"
        assert kw["entity"] == "ent"
        assert kw["tags"] == ["t1"]
        assert kw["notes"] == "n"
        assert kw["group"] == "g"
        assert kw["name"] == "my_exp"
        assert logger._run is mock_run

    def test_run_name_takes_precedence_over_exp_name(self):
        """Test run name takes precedence over exp name."""
        mock_wandb = _make_mock_wandb()
        mock_wandb.init.return_value = _make_mock_run()  # type: ignore[attr-defined]

        logger, _, _ = _build_logger(
            mock_wandb=mock_wandb,
            env={"WANDB_API_KEY": "k"},
            connect_kwargs={"wandb_run_name": "custom-run"},
        )
        exp = _make_experiment(exp_name="my_exp")

        with patch.object(wl_mod, "wandb_module", mock_wandb):
            logger.experiment_started("node1", exp)

        assert mock_wandb.init.call_args[1]["name"] == "custom-run"

    def test_skips_when_not_connected(self):
        """Test skips when not connected."""
        mock_wandb = _make_mock_wandb()
        logger, base, _ = _build_logger(mock_wandb=mock_wandb, env={})
        assert logger._connected is False

        logger.experiment_started("node1", _make_experiment())

        mock_wandb.init.assert_not_called()
        base.experiment_started.assert_called_once()

    def test_skips_when_wandb_not_available(self):
        """Test skips when wandb not available."""
        base = _make_base_mock()
        with patch.dict("os.environ", {}, clear=True), patch.object(wl_mod, "wandb_module", None), patch.object(wl_mod, "RunClass", None):
            logger = WandbLogger(base)

        logger.experiment_started("node1", _make_experiment())
        base.experiment_started.assert_called_once()
        assert _base_logged_at(base, logging.DEBUG)

    def test_skips_if_run_already_active(self):
        """Test skips if run already active."""
        mock_wandb = _make_mock_wandb()
        logger, base, _ = _build_logger(mock_wandb=mock_wandb, env={"WANDB_API_KEY": "k"})
        logger._run = _make_mock_run()

        with patch.object(wl_mod, "wandb_module", mock_wandb):
            logger.experiment_started("node1", _make_experiment())

        mock_wandb.init.assert_not_called()
        base.experiment_started.assert_called_once()

    def test_handles_wandb_init_failure(self):
        """Test handles wandb init failure."""
        mock_wandb = _make_mock_wandb()
        mock_wandb.init.side_effect = RuntimeError("wandb broken")  # type: ignore[attr-defined]

        logger, base, _ = _build_logger(mock_wandb=mock_wandb, env={"WANDB_API_KEY": "k"})
        with patch.object(wl_mod, "wandb_module", mock_wandb):
            logger.experiment_started("node1", _make_experiment())

        assert logger._run is None
        assert _base_logged_at(base, logging.WARNING)
        base.experiment_started.assert_called_once()

    def test_sets_api_key_in_environ(self):
        """Test sets api key in environ."""
        mock_wandb = _make_mock_wandb()
        mock_wandb.init.return_value = _make_mock_run()  # type: ignore[attr-defined]

        logger, _, _ = _build_logger(
            mock_wandb=mock_wandb,
            env={"WANDB_API_KEY": "k"},
            connect_kwargs={"wandb_api_key": "secret"},
        )
        with patch.object(wl_mod, "wandb_module", mock_wandb), patch.dict("os.environ", {}, clear=False) as env:
            logger.experiment_started("node1", _make_experiment())
            assert env.get("WANDB_API_KEY") == "secret"

    def test_forwards_experiment_config(self):
        """Test forwards experiment config."""
        mock_wandb = _make_mock_wandb()
        mock_wandb.init.return_value = _make_mock_run()  # type: ignore[attr-defined]

        logger, _, _ = _build_logger(mock_wandb=mock_wandb, env={"WANDB_API_KEY": "k"})
        exp = _make_experiment(exp_name="cfg", total_rounds=5, learning_rate=0.01)

        with patch.object(wl_mod, "wandb_module", mock_wandb):
            logger.experiment_started("node1", exp)

        cfg = mock_wandb.init.call_args[1]["config"]
        assert cfg["exp_name"] == "cfg"
        assert cfg["total_rounds"] == 5
        assert cfg["learning_rate"] == 0.01

    def test_excludes_none_optionals_from_init_params(self):
        """Test excludes none optionals from init params."""
        mock_wandb = _make_mock_wandb()
        mock_wandb.init.return_value = _make_mock_run()  # type: ignore[attr-defined]

        logger, _, _ = _build_logger(mock_wandb=mock_wandb, env={"WANDB_API_KEY": "k"})
        with patch.object(wl_mod, "wandb_module", mock_wandb):
            logger.experiment_started("node1", _make_experiment())

        kw = mock_wandb.init.call_args[1]
        for key in ("entity", "tags", "notes", "group"):
            assert key not in kw


###
# Tests: on_experiment_change()
###


class TestOnExperimentChange:
    """On Experiment Change tests."""

    def _logger_with_run(self):
        mock_wandb = _make_mock_wandb()
        logger, base, _ = _build_logger(mock_wandb=mock_wandb, env={"WANDB_API_KEY": "k"})
        logger._run = _make_mock_run()
        return logger, base, mock_wandb

    def test_round_change_calls_wandb_log(self):
        """Test round change calls wandb log."""
        logger, _, mock_wandb = self._logger_with_run()
        with patch.object(wl_mod, "wandb_module", mock_wandb):
            logger.on_experiment_change("n1", "round", 3)
        mock_wandb.log.assert_called_once_with({"n1/round": 3})

    def test_config_field_updates_wandb_config(self):
        """Test config field updates wandb config."""
        logger, _, mock_wandb = self._logger_with_run()
        with patch.object(wl_mod, "wandb_module", mock_wandb):
            logger.on_experiment_change("n1", "learning_rate", 0.001)
        mock_wandb.config.update.assert_called_once_with({"learning_rate": 0.001}, allow_val_change=True)

    def test_current_stage_is_skipped(self):
        """Test current stage is skipped."""
        logger, _, mock_wandb = self._logger_with_run()
        with patch.object(wl_mod, "wandb_module", mock_wandb):
            logger.on_experiment_change("n1", "current_stage", "training")
        mock_wandb.log.assert_not_called()
        mock_wandb.config.update.assert_not_called()

    def test_handles_wandb_error(self):
        """Test handles wandb error."""
        logger, base, mock_wandb = self._logger_with_run()
        mock_wandb.log.side_effect = RuntimeError("fail")
        with patch.object(wl_mod, "wandb_module", mock_wandb):
            logger.on_experiment_change("n1", "round", 5)
        assert _base_logged_at(base, logging.WARNING)

    def test_noop_without_run(self):
        """Test noop without run."""
        mock_wandb = _make_mock_wandb()
        logger, base, _ = _build_logger(mock_wandb=mock_wandb, env={"WANDB_API_KEY": "k"})
        assert logger._run is None

        logger.on_experiment_change("n1", "round", 1)

        mock_wandb.log.assert_not_called()
        base.on_experiment_change.assert_called_once()


###
# Tests: log_metric()
###


class TestLogMetric:
    """Log Metric tests."""

    def _logger_with_run(self):
        mock_wandb = _make_mock_wandb()
        logger, base, _ = _build_logger(mock_wandb=mock_wandb, env={"WANDB_API_KEY": "k"})
        logger._run = _make_mock_run()
        return logger, base, mock_wandb

    def test_with_step_passes_round_as_step(self):
        """Test with step passes round as step."""
        logger, base, mock_wandb = self._logger_with_run()
        with patch.object(wl_mod, "wandb_module", mock_wandb):
            logger.log_metric("a1", "loss", 0.5, step=10, round=2)
        mock_wandb.log.assert_called_once_with({"a1/loss": 0.5}, step=2)
        base.log_metric.assert_called_once()

    def test_without_step_logs_without_step_kwarg(self):
        """Test without step logs without step kwarg."""
        logger, base, mock_wandb = self._logger_with_run()
        with patch.object(wl_mod, "wandb_module", mock_wandb):
            logger.log_metric("a1", "acc", 0.95, step=None, round=1)
        mock_wandb.log.assert_called_once_with({"a1/acc": 0.95})

    def test_delegates_when_no_run(self):
        """Test delegates when no run."""
        mock_wandb = _make_mock_wandb()
        logger, base, _ = _build_logger(mock_wandb=mock_wandb, env={"WANDB_API_KEY": "k"})
        assert logger._run is None
        logger.log_metric("a1", "loss", 0.3, step=1, round=0)
        mock_wandb.log.assert_not_called()
        base.log_metric.assert_called_once()

    def test_handles_wandb_log_error(self):
        """Test handles wandb log error."""
        logger, base, mock_wandb = self._logger_with_run()
        mock_wandb.log.side_effect = RuntimeError("network")
        with patch.object(wl_mod, "wandb_module", mock_wandb):
            logger.log_metric("a1", "loss", 0.5, step=1, round=0)
        assert _base_logged_at(base, logging.WARNING)
        base.log_metric.assert_called_once()


###
# Tests: finish()
###


class TestFinish:
    """Finish tests."""

    def test_calls_wandb_finish_and_resets_run(self):
        """Test calls wandb finish and resets run."""
        mock_wandb = _make_mock_wandb()
        logger, base, _ = _build_logger(mock_wandb=mock_wandb, env={"WANDB_API_KEY": "k"})
        logger._run = _make_mock_run()

        with patch.object(wl_mod, "wandb_module", mock_wandb):
            logger.finish()

        mock_wandb.finish.assert_called_once()
        assert logger._run is None
        base.finish.assert_called_once()

    def test_handles_finish_error(self):
        """Test handles finish error."""
        mock_wandb = _make_mock_wandb()
        mock_wandb.finish.side_effect = RuntimeError("fail")  # type: ignore[attr-defined]
        logger, base, _ = _build_logger(mock_wandb=mock_wandb, env={"WANDB_API_KEY": "k"})
        logger._run = _make_mock_run()

        with patch.object(wl_mod, "wandb_module", mock_wandb):
            logger.finish()

        assert logger._run is None
        assert _base_logged_at(base, logging.WARNING)
        base.finish.assert_called_once()

    def test_noop_when_no_run(self):
        """Test noop when no run."""
        mock_wandb = _make_mock_wandb()
        logger, base, _ = _build_logger(mock_wandb=mock_wandb, env={"WANDB_API_KEY": "k"})
        assert logger._run is None

        logger.finish()

        mock_wandb.finish.assert_not_called()
        base.finish.assert_called_once()
