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

from typing import TYPE_CHECKING, Type, Union

from p2pfl.communication.commands.message.metrics_command import MetricsCommand
from p2pfl.management.logger import logger
from p2pfl.stages.stage import EarlyStopException, Stage, check_early_stop

if TYPE_CHECKING:
    from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
    from p2pfl.learning.frameworks.learner import Learner
    from p2pfl.node_state import NodeState

class EvaluateStage(Stage):
    """Train stage."""

    @staticmethod
    async def execute(
        state: NodeState,
        communication_protocol: CommunicationProtocol,
        learner: Learner) -> None:
        """Execute the stage."""
        try:
            # Evaluate and send metrics
            await EvaluateStage.__evaluate(state, learner, communication_protocol)

        except EarlyStopException:
            return None

    @staticmethod
    async def __evaluate(
        state: NodeState,
        learner: Learner,
        communication_protocol: CommunicationProtocol
        ) -> None:
        logger.info(state.addr, "🔬 Evaluating...")
        results = await learner.evaluate()
        logger.info(state.addr, f"📈 Evaluated. Results: {results}")
        # Send metrics
        if len(results) > 0:
            logger.info(state.addr, "📢 Broadcasting metrics.")
            flattened_metrics = [str(item) for pair in results.items() for item in pair]
            await communication_protocol.broadcast(
                communication_protocol.build_msg(
                    MetricsCommand.get_name(),
                    flattened_metrics,
                    round=state.round,
                )
            )
