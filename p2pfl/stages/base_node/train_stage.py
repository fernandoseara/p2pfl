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
"""Train stage."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Set, Type, Union

from p2pfl.communication.commands.message.models_agregated_command import ModelsAggregatedCommand
from p2pfl.communication.commands.weights.partial_model_command import PartialModelCommand
from p2pfl.learning.aggregators.aggregator import NoModelsToAggregateError
from p2pfl.management.logger import logger
from p2pfl.stages.stage import EarlyStopException, Stage, check_early_stop

if TYPE_CHECKING:
    from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
    from p2pfl.learning.aggregators.aggregator import Aggregator
    from p2pfl.learning.frameworks.learner import Learner
    from p2pfl.node_state import NodeState


class TrainStage(Stage):
    """Train stage."""

    @staticmethod
    async def execute(
        state: NodeState,
        learner: Learner,
        ) -> None:
        """Execute the stage."""
        # Train
        logger.info(state.addr, "🏋️‍♀️ Training...")
        await learner.fit()
        logger.info(state.addr, "🎓 Training done.")

        # Add model to the state
        state.add_model(learner.get_model(), source=state.addr)
