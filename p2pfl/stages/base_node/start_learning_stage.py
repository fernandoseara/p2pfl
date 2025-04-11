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

import time
from typing import TYPE_CHECKING, Any

from p2pfl.communication.commands.message.model_initialized_command import ModelInitializedCommand
from p2pfl.communication.commands.message.start_learning_command import StartLearningCommand
from p2pfl.communication.commands.weights.init_model_command import InitModelCommand
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
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
        communication_protocol = node.get_communication_protocol()
        learner = node.get_learner()

        logger.info(node.address, "🚀 Broadcasting start learning...")
        await communication_protocol.broadcast(
            communication_protocol.build_msg(
                StartLearningCommand.get_name(),
                [str(rounds),
                 str(epochs),
                 str(trainset_size),
                 experiment_name]
            )
        )

        # Init
        state.set_experiment(experiment_name, rounds, epochs, trainset_size)
        learner.set_epochs(state.get_experiment().epochs)

        # Wait and gossip model initialization
        logger.info(node.address, "⏳ Waiting initialization.")

        # Communicate Initialization
        await communication_protocol.broadcast(communication_protocol.build_msg(ModelInitializedCommand.get_name()))
