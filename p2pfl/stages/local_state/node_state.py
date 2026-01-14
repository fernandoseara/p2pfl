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
"""Node state."""

from __future__ import annotations

from p2pfl.management.logger import logger
from p2pfl.stages.local_state.experiment import Experiment


class LocalNodeState:
    """
    Class to store the local state of a learning node.

    Peer-related state (nei_status, models_aggregated, train_set_votes) is managed by NetworkState.

    Attributes:
        address: The address of the node.
        status: The status of the node.
        train_set: The train set of the node.
        experiment: The current experiment.

    Args:
        address: The address of the node.

    """

    def __init__(self, address: str) -> None:
        """Initialize the node state."""
        self.address = address
        self.status = "Idle"

        # Train Set
        self.train_set: list[str] = []

        # Actual experiment
        self.experiment: Experiment | None = None

    @property
    def round(self) -> int | None:
        """Get the round."""
        return self.experiment.round if self.experiment is not None else None

    @property
    def total_rounds(self) -> int | None:
        """Get the total rounds."""
        return self.experiment.total_rounds if self.experiment is not None else None

    @property
    def exp_name(self) -> str | None:
        """Get the actual experiment name."""
        return self.experiment.exp_name if self.experiment is not None else None

    @property
    def epochs_per_round(self) -> int | None:
        """Get the number of epochs per round."""
        return self.experiment.epochs_per_round if self.experiment is not None else None

    def get_experiment(self) -> Experiment | None:
        """Get the actual experiment."""
        return self.experiment

    def set_experiment(self, *args, **kwargs) -> None:
        """
        Set a new experiment.

        Args:
            *args: Positional arguments for Experiment (exp_name, total_rounds).
            **kwargs: Additional experiment parameters (epochs_per_round, dataset_name, etc.)

        """
        self.experiment = Experiment(*args, **kwargs)
        logger.experiment_started(self.address, self.experiment)

    def increase_iteration(self) -> None:
        """
        Increase the round number.

        Args:
            exp_name: The name of the experiment.

        Raises:
            ValueError: If the experiment is not initialized.

        """
        if self.experiment is None:
            raise ValueError("Experiment not initialized")

        self.experiment.increase_round()
        logger.experiment_started(self.address, self.experiment)  # TODO: Improve changes on the experiment

    def clear(self) -> None:
        """Clear the state."""
        type(self).__init__(self, self.address)

    def __str__(self) -> str:
        """Return a String representation of the node state."""
        return (
            f"LocalNodeState(address={self.address}, status={self.status}, exp_name={self.exp_name}, "
            f"round={self.round}, total_rounds={self.total_rounds}, "
            f"train_set={self.train_set})"
        )
