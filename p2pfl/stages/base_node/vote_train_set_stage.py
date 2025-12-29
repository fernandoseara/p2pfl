#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2022 Pedro Guijas Bravo.
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
"""Vote Train Set Stage."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from p2pfl.communication.commands.message.vote_train_set_command import VoteTrainSetCommand
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.stages.stage import Stage

if TYPE_CHECKING:
    from p2pfl.node import Node

class VoteTrainSetStage(Stage):
    """Vote Train Set Stage."""

    @staticmethod
    async def execute(node: Node) -> None:
        """Execute the stage."""
        communication_protocol = node.get_communication_protocol()
        state = node.get_local_state()
        generator = node.get_generator()

        # Vote (at least itself)
        candidates = list(communication_protocol.get_neighbors(only_direct=False))
        if node.address not in candidates:
            candidates.append(node.address)
        logger.debug(node.address, f"👨‍🏫 {len(candidates)} candidates to train set")

        # Order candidates to make a deterministic vote (based on the random seed)
        candidates.sort()

        # Send vote
        samples = min(state.get_experiment().trainset_size, len(candidates))
        nodes_voted = generator.sample(candidates, samples)
        weights = [math.floor(generator.randint(0, 1000) / (i + 1)) for i in range(samples)]

        # Add self vote (send it to itself)
        self_vote = list(zip(nodes_voted, weights))
        logger.debug(node.address, f"🪞🗳️ Self Vote: {self_vote}")
        await node.node_workflow.vote(node.address, state.round, self_vote)

        # Convert self vote to a plain list
        votes_list = []
        for peer_voted, weight in self_vote:
            votes_list.append(peer_voted)
            votes_list.append(weight)

        # Send and wait for votes
        logger.info(node.address, "🗳️ Sending train set vote.")
        await communication_protocol.broadcast_gossip(
            communication_protocol.build_msg(
                VoteTrainSetCommand.get_name(),
                votes_list,
                round=state.round,
            )
        )
