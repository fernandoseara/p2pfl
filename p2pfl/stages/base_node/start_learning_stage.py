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
    from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
    from p2pfl.learning.frameworks.learner import Learner
    from p2pfl.node_state import NodeState


class StartLearningStage(Stage):
    """Start learning stage."""

    @staticmethod
    async def execute(
        experiment_name: str,
        rounds: int,
        epochs: int,
        trainset_size: int,
        state: NodeState,
        communication_protocol: CommunicationProtocol,
        learner: Learner,
        ) -> None:
        """Execute the stage."""

        # Init
        state.set_experiment(experiment_name, rounds, epochs, trainset_size)
        learner.set_epochs(state.experiment.epochs)

        # Broadcast start Learning
        logger.info(state.addr, "🚀 Broadcasting start learning...")
        await communication_protocol.broadcast(
            communication_protocol.build_msg(
                StartLearningCommand.get_name(),
                [str(state.experiment.total_rounds),
                 str(state.experiment.epochs),
                 str(state.experiment.trainset_size),
                 state.experiment.exp_name]
            )
        )

        begin = time.time()

        # Wait and gossip model initialization
        logger.info(state.addr, "⏳ Waiting initialization.")

        # Communicate Initialization
        await communication_protocol.broadcast(communication_protocol.build_msg(ModelInitializedCommand.get_name()))
