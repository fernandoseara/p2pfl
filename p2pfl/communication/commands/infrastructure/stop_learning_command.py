#
# This file is part of the p2pfl distribution (see https://github.com/pguijas/p2pfl).
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
"""StopLearning command."""

from __future__ import annotations

from p2pfl.communication.commands.command import Command
from p2pfl.management.logger import logger


class StopLearningCommand(Command):
    """StopLearning command (infrastructure, uses node directly)."""

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "stop_learning"

    async def execute(self, source: str, round: int, **kwargs) -> None:
        """
        Execute the command. Stop learning.

        Args:
            source: The source of the command.
            round: The round of the command.
            **kwargs: The command keyword arguments.

        """
        logger.info(self.node.address, "Stopping learning received")
        if self.node.state.is_learning and self.node.workflow is not None:
            await self.node.workflow.stop()
