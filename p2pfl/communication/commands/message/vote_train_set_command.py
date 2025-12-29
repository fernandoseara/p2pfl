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

"""VoteTrainSetCommand."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from p2pfl.communication.commands.command import Command
from p2pfl.management.logger import logger

if TYPE_CHECKING:  # Only imports the below statements during type checking
    from p2pfl.node import Node


class VoteTrainSetCommand(Command):
    """VoteTrainSetCommand."""

    def __init__(self, node: Node) -> None:
        """Initialize the command."""
        self._node = node

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "vote_train_set"

    async def execute(self, source: str, round: int, *args, **kwargs) -> None:
        """
        Execute the command. Start learning thread.

        Args:
            source: The source of the command.
            round: The round of the command.
            *args: Vote values (pairs of key and values).
            **kwargs: The command keyword arguments.

        """
        # build vote dict
        votes = VoteTrainSetCommand.parse_votes_list(args)

        await self._node.get_event_handler().vote(source, round, votes)

    @staticmethod
    def parse_votes_list(votes_list: list[str]) -> list[tuple[str, int]]:
        """Parse a flat list [peer_voted, weight, peer_voted, weight, ...] into (peer_voted, weight) tuples."""
        if len(votes_list) % 2 != 0:
            raise ValueError("Votes list must contain an even number of elements (peer, weight pairs).")

        parsed_votes = []
        for i in range(0, len(votes_list), 2):
            peer_voted = votes_list[i]
            weight = int(votes_list[i + 1])
            parsed_votes.append((peer_voted, weight))
        return parsed_votes
