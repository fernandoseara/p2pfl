#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2026 Pedro Guijas Bravo.
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

from p2pfl.management.logger import logger
from p2pfl.workflow.engine.stage import Stage

if TYPE_CHECKING:
    from p2pfl.node import Node
    from p2pfl.workflow.async_dfl.workflow import AsyncPeerState


class GossipModelStage(Stage):
    """Gossip model stage."""

    @staticmethod
    async def execute(
        peers: dict[str, AsyncPeerState],
        node: Node,
        candidates: list[str],
    ) -> None:
        """Execute the stage."""
        assert node.workflow is not None
        # Push model to selected neighbors and update push-sum weights
        for neighbor in candidates:
            model_sent = await GossipModelStage.__send_model(neighbor, node)
            # Only send push-sum weight and update time if model was actually sent
            if model_sent:
                await GossipModelStage.__send_push_sum_weight(peers, neighbor, node)
                peers[neighbor].push_time = node.workflow.round or 0

    @staticmethod
    async def __send_model(neighbor: str, node: Node) -> bool:
        """
        Send the model to a neighbor with pre-send confirmation.

        Args:
            neighbor: The neighbor to send the model to.
            node: The node to send the model from.

        Returns:
            True if model was sent, False if recipient declined or error occurred.

        """
        assert node.workflow is not None
        communication_protocol = node.communication_protocol
        model = node.learner.get_model()
        round_num = node.workflow.round or 0
        contributors = model.get_contributors()

        try:
            # Pre-send confirmation: ask neighbor if they want this model
            pre_send_msg = communication_protocol.build_msg(
                "pre_send_model",
                ["model_information_updating"] + contributors,
                round=round_num,
                direct=True,
            )
            response = await communication_protocol.send(neighbor, pre_send_msg, temporal_connection=True)

            if response != "true":
                logger.debug(
                    node.address,
                    f"⏭️ Skipping model send to {neighbor} - recipient declined",
                )
                return False

            # Recipient wants the model, send it
            logger.debug(node.address, f"🗣️ Sending model to {neighbor}")
            await communication_protocol.send(
                nei=neighbor,
                msg=communication_protocol.build_weights(
                    "model_information_updating",
                    round_num,
                    model.encode_parameters(),
                    contributors,
                    model.get_num_samples(),
                ),
            )
            logger.debug(node.address, f"✅ Sent model to {neighbor}")
            return True
        except Exception as e:
            logger.warning(node.address, f"⚠️ Failed to send model to {neighbor}: {e}")
            return False

    @staticmethod
    async def __send_push_sum_weight(peers: dict[str, AsyncPeerState], neighbor: str, node: Node) -> None:
        """
        Send the push-sum weight to a neighbor.

        Args:
            peers: The per-peer state dict.
            neighbor: The neighbor to send the weight to.
            node: The node to send from.

        """
        assert node.workflow is not None
        communication_protocol = node.communication_protocol
        self_peer = peers.get(node.address)
        push_sum_weight = self_peer.push_sum_weight if self_peer else 1.0

        await communication_protocol.send(
            nei=neighbor,
            msg=communication_protocol.build_msg(
                "push_sum_weight_information_updating",
                [push_sum_weight],
                round=node.workflow.round,
            ),
        )
