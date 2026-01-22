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

"""ModelsAggregated command."""

from __future__ import annotations

from p2pfl.communication.commands.command import Command


class ModelsAggregatedCommand(Command):
    """ModelsAggregated command for BasicDFL workflow."""

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "models_aggregated"

    async def execute(self, source: str, round: int, *args, **kwargs) -> None:
        """
        Execute the command.

        Args:
            source: The source of the command.
            round: The round of the command.
            *args: List of models that contribute to the aggregated model.
            **kwargs: The command keyword arguments.

        """
        await self.workflow.aggregated_models_received(source, round, list(args))
