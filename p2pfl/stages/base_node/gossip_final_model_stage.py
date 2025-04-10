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

from typing import TYPE_CHECKING, Any, List, Type, Union

from p2pfl.communication.commands.weights.full_model_command import FullModelCommand
from p2pfl.management.logger import logger
from p2pfl.stages.stage import Stage, check_early_stop

if TYPE_CHECKING:
    from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
    from p2pfl.learning.frameworks.learner import Learner
    from p2pfl.node_state import NodeState

class GossipFinalModelStage(Stage):
    """Gossip model stage."""

    @staticmethod
    async def execute(
        state: NodeState,
        communication_protocol: CommunicationProtocol,
        learner: Learner) -> None:
        """Execute the stage."""
        await GossipFinalModelStage.__gossip_model_difusion(state, communication_protocol, learner)

    @staticmethod
    async def __gossip_model_difusion(
        candidates: list[str],
        state: NodeState,
        communication_protocol: CommunicationProtocol,
        learner: Learner,
    ) -> None:
        logger.info(state.addr, "🗣️ Gossiping aggregated model.")

        def model_fn(node: str) -> Any:
            if state.round is None:
                raise Exception("Round not initialized")
            encoded_model = learner.get_model().encode_parameters()
            return communication_protocol.build_weights(FullModelCommand.get_name(), state.round, encoded_model)

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
