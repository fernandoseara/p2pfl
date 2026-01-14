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

from typing import TYPE_CHECKING, Any

from p2pfl.communication.commands.weights.full_model_command import FullModelCommand
from p2pfl.management.logger import logger
from p2pfl.stages.stage import Stage

if TYPE_CHECKING:
    from p2pfl.node import Node


class GossipFullModelStage(Stage):
    """Gossip model stage."""

    @staticmethod
    async def execute(
        node: Node,
        candidates: list[str],
    ) -> None:
        """Execute the stage."""
        logger.info(node.address, "🗣️ Gossiping full model.")
        await GossipFullModelStage.__gossip_model_difusion(
            candidates=candidates,
            node=node,
        )

    @staticmethod
    async def __gossip_model_difusion(
        candidates: list[str],
        node: Node,
    ) -> None:
        def model_fn(_: str) -> Any:
            if node.get_local_state().get_experiment().round is None:
                raise Exception("Round not initialized")
            encoded_model = node.get_learner().get_model().encode_parameters()
            return node.get_communication_protocol().build_weights(
                FullModelCommand.get_name(), node.get_local_state().get_experiment().round, encoded_model
            )

        for neighbor in candidates:
            payload = model_fn(neighbor)
            try:
                logger.debug(node.address, f"🗣️ Sending model to {neighbor}")
                await node.get_communication_protocol().send(neighbor, payload, temporal_connection=True)
                logger.debug(node.address, f"✅ Sent model to {neighbor}")
            except Exception as e:
                logger.warning(node.address, f"⚠️ Failed to send model to {neighbor}: {e}")
