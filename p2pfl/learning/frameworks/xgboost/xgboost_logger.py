#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2024 Pedro Guijas Bravo.
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

"""XGBoost Callback Logger for P2PFL."""

from typing import Any

import xgboost as xgb

from p2pfl.management.logger import logger as P2PLogger


class XGBoostLogger(xgb.callback.TrainingCallback):
    """
    XGBoost TrainingCallback for Federated Learning logging.

    Logs evaluation metrics per iteration to the P2PFL centralized logger.

    Args:
        addr (str): Address or node identifier for logging.

    Attributes:
        _addr (str): The node address used for logging identification.

    """

    def __init__(self, addr: str) -> None:
        """
        Initialize the XGBoost logger.

        Args:
            addr (str): Address or node identifier for logging.

        """
        self._addr = addr

    def before_training(self, model: xgb.core.Booster) -> None:
        """
        Execute before training starts.

        Args:
            model (xgb.core.Booster): The XGBoost booster model.

        """
        P2PLogger.info(self._addr, "Starting XGBoost training...")

    def after_iteration(
        self,
        model: xgb.core.Booster,
        epoch: int,
        evals_log: dict[str, dict[str, list]],
    ) -> bool:
        """
        Execute after each training iteration.

        Args:
            model (xgb.core.Booster): The XGBoost booster model.
            epoch (int): Current boosting round.
            evals_log (dict[str, dict[str, list]]): Evaluation results as nested dict
                with format: data_name -> metric_name -> list of values.

        Returns:
            bool: False to indicate training should continue.

        """
        for data_name, metrics in evals_log.items():
            for metric_name, history in metrics.items():
                value = history[-1]
                # e.g., 'validation-error'
                full_name = f"{data_name}-{metric_name}"
                P2PLogger.log_metric(self._addr, full_name, value, step=epoch)
        return False

    def after_training(self, model: xgb.core.Booster) -> None:
        """
        Execute after training is finished.

        Args:
            model (xgb.core.Booster): The XGBoost booster model.

        """
        P2PLogger.info(self._addr, "XGBoost training completed.")

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        """
        Log hyperparameters before training.

        Args:
            params (dict[str, Any]): Dictionary of hyperparameters to log.

        """
        pass

    def save(self) -> None:
        """Save logger state (no-op for XGBoostLogger)."""
        pass

    def finalize(self, status: str) -> None:
        """
        Finalize logging with status.

        Args:
            status (str): The final status of training (e.g., 'success' or 'failure').

        """
        P2PLogger.info(self._addr, f"Training finalized with status: {status}")
