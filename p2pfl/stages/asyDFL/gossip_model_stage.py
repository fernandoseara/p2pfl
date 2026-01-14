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
"""Gossip model stage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from p2pfl.communication.commands.message.asyDFL.ctx_info_updating_command import (
    ModelInformationUpdatingCommand,
    PushSumWeightInformationUpdatingCommand,
)
from p2pfl.management.logger import logger
from p2pfl.stages.network_state.async_network_state import AsyncNetworkState

if TYPE_CHECKING:
    from p2pfl.node import Node


class GossipModelStage:
    """Gossip model stage."""

    @staticmethod
    async def execute(
        network_state: AsyncNetworkState,
        node: Node,
        candidates: list[str],
    ) -> None:
        """Execute the stage."""
        # Push model to selected neighbors and update push-sum weights
        for neighbor in candidates:
            await GossipModelStage.__send_model(neighbor, node)
            await GossipModelStage.__send_push_sum_weight(network_state, neighbor, node)
            network_state.update_push_time(neighbor, node.get_local_state().round)

    @staticmethod
    async def __send_model(neighbor: str, node: Node) -> None:
        """
        Send the model to a neighbor.

        Args:
            neighbor: The neighbor to send the model to.
            node: The node to send the model from.

        """
        communication_protocol = node.get_communication_protocol()
        model = node.get_learner().get_model()

        try:
            logger.debug(node.address, f"🗣️ Sending model to {neighbor}")
            await communication_protocol.send(
                nei=neighbor,
                msg=communication_protocol.build_weights(
                    ModelInformationUpdatingCommand.get_name(),
                    node.get_local_state().round,
                    model.encode_parameters(),
                    model.get_contributors(),
                    model.get_num_samples(),
                ),
            )
            logger.debug(node.address, f"✅ Sent model to {neighbor}")
        except Exception as e:
            logger.warning(node.address, f"⚠️ Failed to send model to {neighbor}: {e}")

    @staticmethod
    async def __send_push_sum_weight(network_state: AsyncNetworkState, neighbor: str, node: Node) -> None:
        """
        Send the push-sum weight to a neighbor.

        Args:
            network_state: The async network state.
            neighbor: The neighbor to send the weight to.
            node: The node to send from.

        """
        communication_protocol = node.get_communication_protocol()

        await communication_protocol.send(
            nei=neighbor,
            msg=communication_protocol.build_msg(
                PushSumWeightInformationUpdatingCommand.get_name(),
                [network_state.get_push_sum_weight(node.address)],
                round=node.get_local_state().round,
            ),
        )
