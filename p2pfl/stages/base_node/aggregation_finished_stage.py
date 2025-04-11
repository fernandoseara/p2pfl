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

from typing import TYPE_CHECKING

from p2pfl.communication.commands.message.models_ready_command import ModelsReadyCommand
from p2pfl.management.logger import logger
from p2pfl.stages.stage import Stage

if TYPE_CHECKING:
    from p2pfl.node import Node

class AggregationFinishedStage(Stage):
    """Aggregation Finished stage."""

    @staticmethod
    async def execute(node: Node) -> None:
        """Execute the stage."""
        # Get aggregated model
        logger.debug(
            node.address,
            f"Broadcast aggregation done for round {node.get_local_state().get_experiment().round}",
        )
        # Share that aggregation is done
        await node.get_communication_protocol().broadcast(
            node.get_communication_protocol.build_msg(ModelsReadyCommand.get_name(),
                                                    [],
                                                    round=node.get_local_state().get_experiment().round))
