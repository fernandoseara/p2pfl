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
Weights command for routing model weight transfers to workflows.

This command routes binary weight messages (Weights payload) to the appropriate
handler method on the active workflow.

Protocol mapping:
    Weight name "partial_model" -> workflow.on_weights_partial_model(source, round, weights, contributors, num_samples)
    Weight name "add_model"     -> workflow.on_weights_add_model(source, round, weights, contributors, num_samples)

Note: Weight messages are always direct (not gossiped) due to their size.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from p2pfl.communication.commands.command import Command
from p2pfl.management.logger import logger

if TYPE_CHECKING:
    from p2pfl.node import Node


class WeightsCommand(Command):
    """
    Command that routes weight messages to workflow handlers.

    Routes to `on_weights_<name>` methods on the workflow.
    Used for binary model weight transfers with typed parameters.
    """

    def __init__(self, node: Node, message_name: str) -> None:
        """
        Initialize the command.

        Args:
            node: The node instance for workflow access.
            message_name: The name of the message (used for routing).

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
        *args,
        weights: bytes | None = None,
        contributors: list[str] | None = None,
        num_samples: int | None = None,
        **kwargs,
    ) -> None:
        """
        Execute the command by routing to the appropriate workflow handler.

        Args:
            source: The source of the command.
            round: The round of the command.
            *args: Additional positional arguments (unused).
            weights: Serialized model weights.
            contributors: List of contributors to this model.
            num_samples: Number of samples used to train this model.
            **kwargs: Additional keyword arguments (unused).

        """
        if not self.node.is_learning:
            logger.debug(self.node.address, f"No active workflow for weights {self._message_name}")
            return None

        if weights is None:
            logger.error(self.node.address, f"Missing weights for {self._message_name}")
            return None

        handler_name = f"on_weights_{self._message_name.replace('-', '_')}"
        handler = getattr(self.workflow, handler_name, None)
        if handler is None:
            logger.warning(self.node.address, f"Workflow doesn't implement {handler_name}")
            return None

        return await handler(source, round, weights, contributors, num_samples)
