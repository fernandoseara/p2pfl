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


class GossipPartialModelStage(Stage):
    """GossipPartialModel stage."""

    @staticmethod
    async def execute(
        state: NodeState,
        communication_protocol: CommunicationProtocol,
        learner: Learner,
        aggregator: Aggregator
        ) -> None:
        """Execute the stage."""
        try:
            await GossipPartialModelStage.__gossip_model_aggregation(state, communication_protocol, aggregator)

            # Set aggregated model
            agg_model = aggregator.aggregate(state.models)
            learner.set_model(agg_model)
        except EarlyStopException:
            return None

    @staticmethod
    async def __gossip_model_aggregation(
        candidates,
        state: NodeState,
        communication_protocol: CommunicationProtocol,
    ) -> None:
        """
        Gossip model aggregation.

        CAREFULL:
            - Full connected trainset to increase aggregation speed. On real scenarios, this won't
            be possible, private networks and firewalls.
            - Needed because the trainset can split the networks (and neighbors that are not in the
            trainset won't receive the aggregation).
        """

        def model_fn(node: str) -> Any:
            try:
                model = state.models[node]
            except NoModelsToAggregateError:
                logger.info(state.addr, f"❔ No models to aggregate for {node}.")
                return None
            if state.round is None:
                raise Exception("Round not initialized.")
            return communication_protocol.build_weights(
                PartialModelCommand.get_name(),
                state.round,
                model.encode_parameters(),
                model.get_contributors(),
                model.get_num_samples(),
            )

        # Gossip to eligible neighbors
        logger.debug(state.addr, f"📡 Candidates to gossip to: {candidates}")

        for neighbor in candidates:
            payload = model_fn(neighbor)
            try:
                logger.debug(state.addr, f"🗣️ Sending model to {neighbor}")
                await communication_protocol.send(neighbor, payload, temporal_connection=True)
                logger.debug(state.addr, f"✅ Sent model to {neighbor}")
            except Exception as e:
                logger.warning(state.addr, f"⚠️ Failed to send model to {neighbor}: {e}")
