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

from typing import TYPE_CHECKING
from typing import Dict, List, Optional

from p2pfl.experiment import Experiment
from p2pfl.management.logger import logger

if TYPE_CHECKING:
    from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
    

class LocalNodeState: # A
    """
    Class to store the main state of a learning node.

    Attributes:
        addr: The address of the node.
        status: The status of the node.
        learner: The learner of the node.
        nei_status: The status of the neighbors.
        train_set: The train set of the node.
        train_set_votes: The votes of the train set.

    Args:
        addr: The address of the node.

    """

    def __init__(self, addr: str) -> None:
        """Initialize the node state."""
        self.addr = addr
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

    def get_experiment(self) -> Experiment | None:
        """Get the actual experiment."""
        return self.experiment

    def set_experiment(self, exp_name: str, total_rounds: int, epochs: int = 1, trainset_size: int = 4) -> None:
        """
        Start a new experiment.

        Args:
            exp_name: The name of the experiment.
            total_rounds: The total rounds of the experiment.
            epochs: The number of epochs.
            trainset_size: The size of the train set.

        Raises:
            ValueError: If the experiment is already initialized.

        """
        self.status = "Learning"
        self.experiment = Experiment(exp_name, total_rounds, epochs, trainset_size)
        logger.experiment_started(self.addr, self.experiment)  # TODO: Improve changes on the experiment

    def increase_round(self) -> None:
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
        logger.experiment_started(self.addr, self.experiment)  # TODO: Improve changes on the experiment

    def clear(self) -> None:
        """Clear the state."""
        type(self).__init__(self, self.addr)

    def __str__(self) -> str:
        """Return a String representation of the node state."""
        return (
            f"NodeState(addr={self.addr}, status={self.status}, exp_name={self.exp_name}, "
            f"round={self.round}, total_rounds={self.total_rounds}, "
            f"models={self.models}, nei_status={self.nei_status}, "
            f"train_set={self.train_set}, train_set_votes={self.train_set_votes})"
        )
