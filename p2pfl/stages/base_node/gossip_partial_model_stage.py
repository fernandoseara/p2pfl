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
    from p2pfl.node import Node

class GossipPartialModelStage(Stage):
    """GossipPartialModel stage."""

    @staticmethod
    async def execute(
        node: Node,
        candidates: list[str]
        ) -> None:
        """Execute the stage."""
        try:
            # Communicate Aggregation
            await node.get_communication_protocol().broadcast(
                node.get_communication_protocol().build_msg(
                    ModelsAggregatedCommand.get_name(),
                    node.get_network_state().get_all_contributors(),
                    round=node.get_local_state().round,
                )
            )
            await GossipPartialModelStage.__gossip_model_aggregation(node=node, candidates=candidates)
        except EarlyStopException:
            return None

    @staticmethod
    async def __gossip_model_aggregation(
        node: Node,
        candidates: list[str],
    ) -> None:
        """
        Gossip model aggregation.

        CAREFULL:
            - Full connected trainset to increase aggregation speed. On real scenarios, this won't
            be possible, private networks and firewalls.
            - Needed because the trainset can split the networks (and neighbors that are not in the
            trainset won't receive the aggregation).
        """

        def model_fn(n: str) -> Any:
            model = node.get_network_state().get_model(n)

            if model is None:
                logger.info(node.address, f"❔ No models to aggregate for {node.address}.")
                return None

            return node.get_communication_protocol().build_weights(
                PartialModelCommand.get_name(),
                node.get_local_state().get_experiment().round,
                model.encode_parameters(),
                model.get_contributors(),
                model.get_num_samples(),
            )

        # Gossip to eligible neighbors
        logger.debug(node.address, f"📡 Candidates to gossip to: {candidates}")

        for neighbor in candidates:
            payload = model_fn(neighbor)
            try:
                logger.debug(node.address, f"🗣️ Sending model to {neighbor}")
                await node.get_communication_protocol().send(neighbor, payload, temporal_connection=True)
                logger.debug(node.address, f"✅ Sent model to {neighbor}")
            except Exception as e:
                logger.warning(node.address, f"⚠️ Failed to send model to {neighbor}: {e}")
