#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2022 Pedro Guijas Bravo.
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
"""Start learning stage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from p2pfl.communication.commands.message.node_initialized_command import NodeInitializedCommand
from p2pfl.communication.commands.message.start_learning_command import StartLearningCommand
from p2pfl.management.logger import logger
from p2pfl.stages.stage import Stage

if TYPE_CHECKING:
    from p2pfl.node import Node


class StartLearningStage(Stage):
    """Start learning stage."""

    @staticmethod
    async def execute(
        experiment_name: str,
        rounds: int,
        epochs: int,
        trainset_size: int,
        node: Node,
        ) -> None:
        """Execute the stage."""
        state = node.get_local_state()
        learner = node.get_learner()

        node.get_network_state().add_peer(node.address)
        state.set_experiment(experiment_name, rounds, epochs, trainset_size)
        learner.set_epochs(state.get_experiment().epochs)
