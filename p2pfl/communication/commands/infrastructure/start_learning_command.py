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
"""StartLearning command."""

from __future__ import annotations

from p2pfl.communication.commands.command import Command
from p2pfl.exceptions import NodeRunningException
from p2pfl.management.logger import logger
from p2pfl.workflow.factory import WorkflowType


class StartLearningCommand(Command):
    """StartLearning command (infrastructure, uses node directly)."""

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "start_learning"

    async def execute(
        self,
        source: str,
        round: int,
        learning_rounds: int | None = None,
        learning_epochs: int | None = None,
        trainset_size: int | None = None,
        experiment_name: str | None = None,
        workflow: str | None = None,
        **kwargs,
    ) -> None:
        """
        Execute the command. Start learning.

        Args:
            source: The source of the command.
            round: The round of the command.
            learning_rounds: The number of learning rounds.
            learning_epochs: The number of learning epochs.
            trainset_size: The size of the trainset.
            experiment_name: The name of the experiment.
            workflow: The workflow type to use.
            **kwargs: The command keyword arguments.

        """
        if learning_rounds is None or learning_epochs is None or trainset_size is None or experiment_name is None or workflow is None:
            raise ValueError("Learning rounds and epochs, trainset size, experiment name, and workflow are required")

        try:
            await self.node.peer_learning_initiated(
                workflow_type=WorkflowType(workflow),
                experiment_name=experiment_name,
                rounds=int(learning_rounds),
                epochs=int(learning_epochs),
                trainset_size=int(trainset_size),
                source=source,
            )

        except NodeRunningException as e:
            logger.debug(self.node.address, str(e))
