#
# This file is part of the p2pfl distribution
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
"""Tests for P2PFLogger covering missed lines."""

import logging
from unittest.mock import MagicMock

import pytest

from p2pfl.management.logger.logger import (
    BLUE,
    GREEN,
    RED,
    RESET,
    YELLOW,
    ColoredFormatter,
    NodeNotRegistered,
    P2PFLogger,
)
from p2pfl.settings import Settings
from p2pfl.workflow.engine.experiment import Experiment


@pytest.fixture()
def fresh_logger():
    """Create a fresh P2PFLogger instance, cleaning up the singleton logger after."""
    # Remove any existing handlers from the 'p2pfl' logger so we can re-init
    py_logger = logging.getLogger("p2pfl")
    old_handlers = py_logger.handlers[:]
    py_logger.handlers.clear()
    lgr = P2PFLogger(disable_locks=True)
    yield lgr
    lgr.cleanup()
    # Restore original handlers
    py_logger.handlers = old_handlers


class TestColoredFormatter:
    """ColoredFormatter tests."""

    def test_debug_level_color(self):
        """Debug level uses blue color."""
        fmt = ColoredFormatter("%(levelname)s %(message)s")
        record = logging.LogRecord("test", logging.DEBUG, "", 0, "msg", (), None)
        result = fmt.format(record)
        assert BLUE in result
        assert RESET in result

    def test_warning_level_color(self):
        """Warning level uses yellow color."""
        fmt = ColoredFormatter("%(levelname)s %(message)s")
        record = logging.LogRecord("test", logging.WARNING, "", 0, "msg", (), None)
        result = fmt.format(record)
        assert YELLOW in result

    def test_error_level_color(self):
        """Error level uses red color."""
        fmt = ColoredFormatter("%(levelname)s %(message)s")
        record = logging.LogRecord("test", logging.ERROR, "", 0, "msg", (), None)
        result = fmt.format(record)
        assert RED in result

    def test_critical_level_color(self):
        """Critical level uses red color."""
        fmt = ColoredFormatter("%(levelname)s %(message)s")
        record = logging.LogRecord("test", logging.CRITICAL, "", 0, "msg", (), None)
        result = fmt.format(record)
        assert RED in result

    def test_info_level_color(self):
        """Info level uses green color."""
        fmt = ColoredFormatter("%(levelname)s %(message)s")
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        result = fmt.format(record)
        assert GREEN in result


class TestP2PFLoggerInit:
    """P2PFLogger initialization tests."""

    def test_double_init_raises(self, fresh_logger):
        """Re-creating a logger when handlers already exist should raise."""
        with pytest.raises(Exception, match="Logger already initialized"):
            P2PFLogger(disable_locks=True)


class TestP2PFLoggerConnect:
    """P2PFLogger connect tests."""

    def test_connect_is_noop(self, fresh_logger):
        """connect() should do nothing by default."""
        fresh_logger.connect(foo="bar")  # should not raise


class TestP2PFLoggerFinish:
    """P2PFLogger finish tests."""

    def test_finish_is_noop(self, fresh_logger):
        """finish() should do nothing by default."""
        fresh_logger.finish()  # should not raise


class TestP2PFLoggerCleanup:
    """P2PFLogger cleanup tests."""

    def test_cleanup_removes_handlers_and_nodes(self, fresh_logger):
        """Cleanup removes handlers and registered nodes."""
        fresh_logger.register_node("n1")
        assert "n1" in fresh_logger.get_nodes()
        fresh_logger.cleanup()
        assert len(fresh_logger.get_nodes()) == 0
        assert len(fresh_logger._logger.handlers) == 0


class TestP2PFLoggerLevels:
    """P2PFLogger level tests."""

    def test_set_level_string(self, fresh_logger):
        """Set level from string."""
        fresh_logger.set_level("DEBUG")
        assert fresh_logger.get_level() == logging.DEBUG

    def test_set_level_int(self, fresh_logger):
        """Set level from int."""
        fresh_logger.set_level(logging.WARNING)
        assert fresh_logger.get_level() == logging.WARNING

    def test_get_level_name(self, fresh_logger):
        """Get level name from int."""
        assert fresh_logger.get_level_name(logging.ERROR) == "ERROR"


class TestP2PFLoggerLogMethods:
    """P2PFLogger log method tests."""

    def test_critical(self, fresh_logger):
        """Critical log does not raise."""
        fresh_logger.register_node("n1")
        # Should not raise, just logs
        fresh_logger.critical("n1", "critical msg")

    def test_error(self, fresh_logger):
        """Error log does not raise."""
        fresh_logger.register_node("n1")
        fresh_logger.error("n1", "error msg")

    def test_log_invalid_level_raises(self, fresh_logger):
        """Invalid log level raises ValueError."""
        with pytest.raises(ValueError, match="Invalid level"):
            fresh_logger.log(999, "node", "msg")

    def test_log_with_round_and_stage_context(self, fresh_logger):
        """log() builds context string from node round/stage."""
        fresh_logger.register_node("n1")
        fresh_logger._nodes["n1"]["round"] = 3
        fresh_logger._nodes["n1"]["stage"] = "aggregation"
        fresh_logger.set_level("DEBUG")
        # Should not raise; exercises context building
        fresh_logger.debug("n1", "test message")


class TestP2PFLoggerMetrics:
    """P2PFLogger metric tests."""

    def test_log_metric_node_not_registered_skips(self, fresh_logger):
        """log_metric with unregistered node should return silently."""
        fresh_logger.log_metric("unknown_node", "loss", 0.5, step=1, round=0)

    def test_log_metric_no_round_raises(self, fresh_logger):
        """log_metric without round raises."""
        fresh_logger.register_node("n1")
        exp = Experiment(exp_name="test", total_rounds=1)
        fresh_logger.experiment_started("n1", exp)
        # Remove the round so it's None
        fresh_logger._nodes["n1"].pop("round", None)
        with pytest.raises(Exception, match="No round provided"):
            fresh_logger.log_metric("n1", "loss", 0.5)

    def test_log_metric_no_exp_name_raises(self, fresh_logger):
        """log_metric without experiment name raises."""
        fresh_logger.register_node("n1")
        exp = MagicMock()
        exp.exp_name = None
        fresh_logger._nodes["n1"]["Experiment"] = exp
        fresh_logger._nodes["n1"]["round"] = 0
        with pytest.raises(Exception, match="No experiment name"):
            fresh_logger.log_metric("n1", "loss", 0.5)

    def test_get_local_and_global_logs(self, fresh_logger):
        """Local and global logs are stored correctly."""
        fresh_logger.register_node("n1")
        exp = Experiment(exp_name="exp1", total_rounds=2)
        fresh_logger.experiment_started("n1", exp)
        # Local (with step)
        fresh_logger.log_metric("n1", "loss", 0.5, step=1, round=0)
        # Global (without step)
        fresh_logger.log_metric("n1", "acc", 0.9, round=0)
        local = fresh_logger.get_local_logs()
        assert "exp1" in local
        global_ = fresh_logger.get_global_logs()
        assert "exp1" in global_


class TestP2PFLoggerNodeRegistration:
    """P2PFLogger node registration tests."""

    def test_register_duplicate_raises(self, fresh_logger):
        """Registering duplicate node raises."""
        fresh_logger.register_node("n1")
        with pytest.raises(Exception, match="already registered"):
            fresh_logger.register_node("n1")

    def test_unregister_nonexistent_warns(self, fresh_logger):
        """Unregistering a non-registered node should log a warning, not raise."""
        fresh_logger.unregister_node("ghost_node")  # no raise


class TestP2PFLoggerExperimentLifecycle:
    """P2PFLogger experiment lifecycle tests."""

    def test_experiment_started_unregistered_raises(self, fresh_logger):
        """Starting experiment on unregistered node raises."""
        exp = Experiment(exp_name="e1", total_rounds=1)
        with pytest.raises(NodeNotRegistered):
            fresh_logger.experiment_started("ghost", exp)

    def test_experiment_ended_clears_experiment_state(self, fresh_logger):
        """Ending experiment clears experiment state."""
        fresh_logger.register_node("n1")
        exp = Experiment(exp_name="e1", total_rounds=1)
        fresh_logger.experiment_started("n1", exp)
        assert "Experiment" in fresh_logger._nodes["n1"]
        assert fresh_logger._nodes["n1"]["round"] == 0
        fresh_logger.experiment_ended("n1", exp, "finished")
        assert "Experiment" not in fresh_logger._nodes["n1"]
        assert "round" not in fresh_logger._nodes["n1"]

    def test_on_experiment_change_updates_round_and_stage(self, fresh_logger):
        """Experiment change updates round and stage."""
        fresh_logger.register_node("n1")
        fresh_logger.on_experiment_change("n1", "round", 5)
        assert fresh_logger._nodes["n1"]["round"] == 5
        fresh_logger.on_experiment_change("n1", "current_stage", "eval")
        assert fresh_logger._nodes["n1"]["stage"] == "eval"

    def test_on_experiment_change_unregistered_noop(self, fresh_logger):
        """on_experiment_change for unregistered node should silently return."""
        fresh_logger.on_experiment_change("ghost", "round", 1)


class TestP2PFLoggerGetNodes:
    """P2PFLogger get_nodes tests."""

    def test_get_nodes(self, fresh_logger):
        """Get nodes returns all registered nodes."""
        fresh_logger.register_node("a")
        fresh_logger.register_node("b")
        assert set(fresh_logger.get_nodes().keys()) == {"a", "b"}


class TestP2PFLoggerCommunication:
    """P2PFLogger communication log tests."""

    def test_log_communication_disabled(self, fresh_logger):
        """When LOG_COMMUNICATION is False, log_communication returns early."""
        original = Settings.general.LOG_COMMUNICATION
        Settings.general.LOG_COMMUNICATION = False
        try:
            fresh_logger.log_communication("n1", "sent", "hello", "n2", "message", 100)
            # Should have no messages stored
            assert len(fresh_logger.get_messages()) == 0
        finally:
            Settings.general.LOG_COMMUNICATION = original

    def test_log_communication_resolves_round_from_node(self, fresh_logger):
        """When round_num is None, it should pick round from node state."""
        fresh_logger.register_node("n1")
        fresh_logger._nodes["n1"]["round"] = 7
        fresh_logger.set_level("DEBUG")
        fresh_logger.log_communication("n1", "sent", "cmd", "n2", "message", 50)
        msgs = fresh_logger.get_messages()
        assert len(msgs) == 1
        assert msgs[0]["round"] == 7

    def test_log_communication_negative_round_resolves_from_node(self, fresh_logger):
        """When round_num is negative, it should try to resolve from node."""
        fresh_logger.register_node("n1")
        fresh_logger._nodes["n1"]["round"] = 3
        fresh_logger.set_level("DEBUG")
        fresh_logger.log_communication("n1", "sent", "cmd", "n2", "message", 50, round_num=-1)
        msgs = fresh_logger.get_messages()
        assert msgs[0]["round"] == 3


class TestP2PFLoggerGetMessages:
    """P2PFLogger get_messages tests."""

    def test_get_messages_invalid_direction_raises(self, fresh_logger):
        """Invalid direction raises ValueError."""
        with pytest.raises(ValueError, match="Invalid direction"):
            fresh_logger.get_messages(direction="invalid")

    def test_get_messages_filter_by_direction(self, fresh_logger):
        """Messages can be filtered by direction."""
        fresh_logger.set_level("DEBUG")
        fresh_logger.log_communication("n1", "sent", "cmd", "n2", "message", 10)
        fresh_logger.log_communication("n1", "received", "cmd", "n2", "message", 10)
        assert len(fresh_logger.get_messages(direction="sent")) == 1
        assert len(fresh_logger.get_messages(direction="received")) == 1
        assert len(fresh_logger.get_messages(direction="all")) == 2


class TestP2PFLoggerSystemMetrics:
    """P2PFLogger system metrics tests."""

    def test_get_system_metrics(self, fresh_logger):
        """Get system metrics returns a dict."""
        metrics = fresh_logger.get_system_metrics()
        assert isinstance(metrics, dict)


class TestP2PFLoggerReset:
    """P2PFLogger reset tests."""

    def test_reset_clears_state(self, fresh_logger):
        """Reset clears all state."""
        fresh_logger.register_node("n1")
        fresh_logger.set_level("DEBUG")
        fresh_logger.log_communication("n1", "sent", "cmd", "n2", "message", 10)
        fresh_logger.reset()
        assert len(fresh_logger.get_nodes()) == 0
        assert len(fresh_logger.get_messages()) == 0
        assert fresh_logger.get_local_logs() == {}
        assert fresh_logger.get_global_logs() == {}
