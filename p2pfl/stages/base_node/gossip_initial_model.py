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

from p2pfl.communication.commands.weights.init_model_command import InitModelCommand
from p2pfl.management.logger import logger
from p2pfl.stages.stage import Stage, check_early_stop

if TYPE_CHECKING:
    from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
    from p2pfl.learning.frameworks.learner import Learner
    from p2pfl.node_state import NodeState

class GossipInitialModelStage(Stage):
    """Gossip initial model stage."""

    @staticmethod
    async def execute(
        state: NodeState,
        communication_protocol: CommunicationProtocol,
        learner: Learner) -> None:
        """Execute the stage."""
        logger.info(state.addr, "🗣️ Gossiping model initialization.")
        await GossipInitialModelStage.__gossip_model(state, communication_protocol, learner)

    @staticmethod
    async def __gossip_model(
        state: NodeState,
        communication_protocol: CommunicationProtocol,
        learner: Learner,
    ) -> None:
        def early_stopping_fn():
            return state.round is None

        # Wait a model (init or aggregated)
        def candidate_condition(node: str) -> bool:
            return node not in state.nei_status

        def get_candidates_fn() -> list[str]:
            return [n for n in communication_protocol.get_neighbors(only_direct=True) if candidate_condition(n)]

        def status_fn() -> Any:
            return get_candidates_fn()

        def model_fn(_: str) -> Any:
            if state.round is None:
                raise Exception("Round not initialized.")
            encoded_model = learner.get_model().encode_parameters()
            return communication_protocol.build_weights(InitModelCommand.get_name(), state.round, encoded_model)

        # Gossip
        await communication_protocol.gossip_weights(
            early_stopping_fn,
            get_candidates_fn,
            status_fn,
            model_fn,
        )
