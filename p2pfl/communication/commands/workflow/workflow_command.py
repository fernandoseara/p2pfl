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
Workflow command for routing messages to workflow handlers.

Created automatically by the Node from the workflow's message declarations.
At execution time, resolves the handler on the current active workflow.
The server distinguishes message vs weight payloads via protobuf ``oneof``
and calls ``execute()`` with the appropriate arguments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from p2pfl.communication.commands.command import Command
from p2pfl.management.logger import logger

if TYPE_CHECKING:
    from p2pfl.node import Node


class WorkflowCommand(Command):
    """
    Command that routes workflow messages to their handlers.

    Handles both regular messages (string args) and weight transfers
    (binary data). The server determines the payload type from the
    protobuf ``oneof`` and passes the appropriate arguments.
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
    ) -> str | None:
        """
        Execute the command by resolving and calling the workflow handler.

        Args:
            source: The source of the command.
            round: The round of the command.
            *args: String arguments from gossip/direct messages.
            weights: Serialized model weights (for weight transfers).
            contributors: List of contributors to this model.
            num_samples: Number of samples used to train this model.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Optional response string (for direct messages).

        """
        if not self.node.state.is_learning:
            logger.debug(self.node.address, f"No active workflow for '{self._message_name}'")
            return None

        workflow = self.workflow
        if workflow is None:
            return None
        handlers = [
            cb
            for cb, entry in workflow._handlers.get(self._message_name, [])
            if entry.during is None or workflow.current_stage_name in entry.during
        ]
        if not handlers:
            logger.debug(self.node.address, f"No active handler for '{self._message_name}' in stage '{workflow.current_stage_name}'")
            return None

        result = None
        if weights is not None:
            for handler in handlers:
                await handler(source, round, weights, contributors, num_samples)
        else:
            for handler in handlers:
                r = await handler(source, round, *args)
                if r is not None:
                    result = r
        return result
