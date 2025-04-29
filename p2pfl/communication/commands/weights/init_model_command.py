#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2024 Pedro Guijas Bravo.
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

"""InitModel command."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from p2pfl.communication.commands.command import Command
from p2pfl.management.logger import logger

if TYPE_CHECKING:  # Only imports the below statements during type checking
    from p2pfl.node import Node


class InitModelCommand(Command):
    """InitModelCommand."""

    def __init__(self, node: Node) -> None:
        """Initialize InitModelCommand."""
        self.__node = node

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "init_model"

    async def execute(
        self,
        source: str,
        round: int,
        weights: Optional[bytes] = None,
        **kwargs,
    ) -> None:
        """Execute the command."""
        if weights is None:
            logger.error(self.__node.address, "Invalid InitModelCommand message")
            return

        await self.__node.learning_workflow.initial_model_received(source, weights)
