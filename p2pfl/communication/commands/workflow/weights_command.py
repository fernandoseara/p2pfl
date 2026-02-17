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
"""
Weights command for routing model weight transfers to workflow handlers.

Created automatically by the Workflow from its ``get_message_registry()``.
At execution time, resolves the handler on the current active workflow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from p2pfl.communication.commands.command import Command
from p2pfl.management.logger import logger

if TYPE_CHECKING:
    from p2pfl.node import Node


class WeightsCommand(Command):
    """
    Command that routes weight messages to a workflow handler.

    At execution time, resolves the handler by looking up the command name
    in the active workflow's ``get_message_registry()``.
    """

    def __init__(self, node: Node, message_name: str) -> None:
        """
        Initialize the command.

        Args:
            node: The node instance.
            message_name: The command name used for message routing.

        """
        super().__init__(node)
        self._message_name = message_name

    def get_name(self) -> str:  # type: ignore[override]
        """Get the command name."""
        return self._message_name

    async def execute(
        self,
        source: str,
        round: int,
        *args: Any,
        weights: bytes | None = None,
        contributors: list[str] | None = None,
        num_samples: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Execute the command by resolving and calling the workflow handler.

        Args:
            source: The source of the command.
            round: The round of the command.
            *args: Additional positional arguments (unused).
            weights: Serialized model weights.
            contributors: List of contributors to this model.
            num_samples: Number of samples used to train this model.
            **kwargs: Additional keyword arguments (unused).

        """
        if not self.node.state.is_learning:
            logger.debug(self.node.address, f"No active workflow for weights '{self._message_name}'")
            return None

        if weights is None:
            logger.error(self.node.address, f"Missing weights for '{self._message_name}'")
            return None

        workflow = self.workflow
        entry = workflow.get_message_registry().get(self._message_name)
        if entry is None:
            logger.warning(self.node.address, f"Workflow has no handler for '{self._message_name}'")
            return None

        handler = getattr(workflow, entry.method_name)
        await handler(source, round, weights, contributors, num_samples)
