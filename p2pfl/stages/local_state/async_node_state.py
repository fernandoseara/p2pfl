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

from p2pfl.management.logger import logger
from p2pfl.stages.local_state.node_state import LocalNodeState

if TYPE_CHECKING:
    pass


class AsyncLocalNodeState(LocalNodeState):
    """
    Class to store the main state of a learning node.

    Attributes:
        addr: The address of the node.
        learner: The learner of the node.
        nei_status: The status of the neighbors.
        train_set: The train set of the node.
        train_set_votes: The votes of the train set.

    Args:
        addr: The address of the node.

    """

    def __init__(self, address: str) -> None:
        """Initialize the node state."""
        super().__init__(address)

    def increase_iteration(self) -> None:
        """
        Increase the iteration number.

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
            f"NodeState(addr={self.address}, status={self.status}, exp_name={self.exp_name}, "
            f"round={self.round}, total_rounds={self.total_rounds}, "
            f"train_set={self.train_set})"
        )
