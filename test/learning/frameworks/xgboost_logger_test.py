"""Tests for XGBoostLogger callback."""

from unittest.mock import MagicMock, call, patch

import pytest

xgb = pytest.importorskip("xgboost", reason="XGBoost not available or missing OpenMP dependency")

from p2pfl.learning.frameworks.xgboost.xgboost_logger import XGBoostLogger  # noqa: E402

###
# Unit tests
###


class TestXGBoostLoggerCallbacks:
    """Test each callback method in isolation with a mocked P2PFL logger."""

    @patch("p2pfl.learning.frameworks.xgboost.xgboost_logger.P2PLogger")
    def test_before_training_logs_info(self, mock_logger):
        """Before_training emits an info log with the node address."""
        cb = XGBoostLogger("node-1")
        cb.before_training(MagicMock())
        mock_logger.info.assert_called_once_with("node-1", "Starting XGBoost training...")

    @patch("p2pfl.learning.frameworks.xgboost.xgboost_logger.P2PLogger")
    def test_after_training_logs_info(self, mock_logger):
        """After_training emits a completion log."""
        cb = XGBoostLogger("node-1")
        cb.after_training(MagicMock())
        mock_logger.info.assert_called_once_with("node-1", "XGBoost training completed.")

    @patch("p2pfl.learning.frameworks.xgboost.xgboost_logger.P2PLogger")
    def test_finalize_logs_status(self, mock_logger):
        """Finalize logs the final status string."""
        cb = XGBoostLogger("node-1")
        cb.finalize("success")
        mock_logger.info.assert_called_once_with("node-1", "Training finalized with status: success")

    @patch("p2pfl.learning.frameworks.xgboost.xgboost_logger.P2PLogger")
    def test_after_iteration_single_metric(self, mock_logger):
        """After_iteration logs a single metric with the correct full name and step."""
        cb = XGBoostLogger("node-2")
        evals_log = {"validation": {"error": [0.3, 0.2, 0.1]}}
        result = cb.after_iteration(MagicMock(), epoch=2, evals_log=evals_log)
        assert result is False  # Should return False to continue training
        mock_logger.log_metric.assert_called_once_with("node-2", "validation-error", 0.1, step=2)

    @patch("p2pfl.learning.frameworks.xgboost.xgboost_logger.P2PLogger")
    def test_after_iteration_multiple_metrics(self, mock_logger):
        """After_iteration logs every metric across all data names."""
        cb = XGBoostLogger("node-3")
        evals_log = {
            "train": {"rmse": [0.5, 0.4], "mae": [0.6, 0.5]},
            "eval": {"rmse": [0.7, 0.6]},
        }
        cb.after_iteration(MagicMock(), epoch=1, evals_log=evals_log)
        expected_calls = [
            call("node-3", "train-rmse", 0.4, step=1),
            call("node-3", "train-mae", 0.5, step=1),
            call("node-3", "eval-rmse", 0.6, step=1),
        ]
        mock_logger.log_metric.assert_has_calls(expected_calls, any_order=True)
        assert mock_logger.log_metric.call_count == 3

    @patch("p2pfl.learning.frameworks.xgboost.xgboost_logger.P2PLogger")
    def test_after_iteration_empty_evals_log(self, mock_logger):
        """After_iteration with empty evals_log does not log anything."""
        cb = XGBoostLogger("node-4")
        result = cb.after_iteration(MagicMock(), epoch=0, evals_log={})
        assert result is False
        mock_logger.log_metric.assert_not_called()


###
# Integration: XGBoostLogger with real XGBoost training
###


class TestXGBoostLoggerIntegration:
    """Run a small XGBoost training loop and verify the logger fires correctly."""

    @patch("p2pfl.learning.frameworks.xgboost.xgboost_logger.P2PLogger")
    def test_before_training_returns_model(self, mock_logger):
        """before_training returns the booster so xgb.train works correctly."""
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=50, n_features=5, n_classes=2, random_state=42)
        dtrain = xgb.DMatrix(X, label=y)

        cb = XGBoostLogger("integration-node")
        xgb.train(
            {"objective": "binary:logistic", "eval_metric": "error", "verbosity": 0},
            dtrain,
            num_boost_round=3,
            evals=[(dtrain, "train")],
            callbacks=[cb],
        )
        mock_logger.info.assert_any_call("integration-node", "Starting XGBoost training...")

    @patch("p2pfl.learning.frameworks.xgboost.xgboost_logger.P2PLogger")
    def test_callbacks_fire_during_manual_loop(self, mock_logger):
        """Manually invoke callbacks and verify the logger fires correctly."""
        cb = XGBoostLogger("integration-node")
        booster = MagicMock()

        cb.before_training(booster)
        mock_logger.info.assert_called_with("integration-node", "Starting XGBoost training...")

        # Simulate 3 iterations with increasing history
        for epoch in range(3):
            evals_log = {"train": {"error": list(range(epoch + 1))}}
            result = cb.after_iteration(booster, epoch=epoch, evals_log=evals_log)
            assert result is False

        assert mock_logger.log_metric.call_count == 3
        # Verify each call logged "train-error" with the latest value
        for i, c in enumerate(mock_logger.log_metric.call_args_list):
            assert c[0][1] == "train-error"
            assert c == call("integration-node", "train-error", i, step=i)

        cb.after_training(booster)
        assert any(c == call("integration-node", "XGBoost training completed.") for c in mock_logger.info.call_args_list)
