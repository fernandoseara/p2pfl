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
Message command for routing gossip/direct messages to workflows.

This command routes string-based messages (GossipMessage or DirectMessage payloads)
to the appropriate handler method on the active workflow.

Protocol mapping:
    Message name "node_initialized" -> workflow.on_message_node_initialized(source, round, *args)
    Message name "vote-train-set"   -> workflow.on_message_vote_train_set(source, round, *args)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from p2pfl.communication.commands.command import Command
from p2pfl.management.logger import logger

if TYPE_CHECKING:
    from p2pfl.node import Node


class MessageCommand(Command):
    """
    Command that routes gossip/direct messages to workflow handlers.

    Routes to `on_message_<name>` methods on the workflow.
    Used for control signals, votes, notifications - anything with string arguments.
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

    async def execute(self, source: str, round: int, *args: str, **kwargs) -> None:
        """
        Execute the command by routing to the appropriate workflow handler.

        Args:
            source: The source of the command.
            round: The round of the command.
            *args: String arguments from the message.
            **kwargs: Unused (for compatibility).

        """
        if not self.node.is_learning:
            logger.debug(self.node.address, f"No active workflow for message {self._message_name}")
            return None

        handler_name = f"on_message_{self._message_name.replace('-', '_')}"
        handler = getattr(self.workflow, handler_name, None)
        if handler is None:
            logger.warning(self.node.address, f"Workflow doesn't implement {handler_name}")
            return None
        return await handler(source, round, *args)
