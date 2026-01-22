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

"""Command interface."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from p2pfl.node import Node


class Command(abc.ABC):
    """Base class for all commands."""

    def __init__(self, node: Node | None = None) -> None:
        """
        Initialize the command.

        Args:
            node: The node instance for workflow access. Optional for infrastructure commands.

        """
        self._node = node

    @property
    def node(self) -> Node:
        """Get the node instance. Raises if not configured."""
        if self._node is None:
            raise RuntimeError("This command requires a node but none was provided")
        return self._node

    @property
    def workflow(self) -> Any:
        """Get the workflow."""
        return self.node.get_learning_workflow()

    @staticmethod
    @abc.abstractmethod
    def get_name() -> str:
        """Get the command name."""
        ...

    @abc.abstractmethod
    async def execute(self, source: str, round: int, *args, **kwargs) -> str | None:
        """
        Execute the command.

        Args:
            source: The source of the command.
            round: The round of the command.
            *args: Additional positional arguments for subclasses.
            **kwargs: Additional keyword arguments for subclasses.

        Returns:
            Optional response string.

        """
        ...
