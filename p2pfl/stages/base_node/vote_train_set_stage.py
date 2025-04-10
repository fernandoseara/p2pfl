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
import random
import time
from typing import TYPE_CHECKING, Type, Union

from p2pfl.communication.commands.message.vote_train_set_command import VoteTrainSetCommand
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.stages.stage import EarlyStopException, Stage, check_early_stop

if TYPE_CHECKING:
    from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
    from p2pfl.learning.aggregators.aggregator import Aggregator
    from p2pfl.node_state import NodeState

class VoteTrainSetStage(Stage):
    """Vote Train Set Stage."""

    @staticmethod
    async def execute(
        state: NodeState,
        communication_protocol: CommunicationProtocol,
        aggregator: Aggregator,
        generator: random.Random,
        ) -> None:
        """Execute the stage."""
        try:
            # Vote
            await VoteTrainSetStage.__vote(state, communication_protocol, generator)
        except EarlyStopException:
            return

    @staticmethod
    async def __vote(state: NodeState, communication_protocol: CommunicationProtocol, generator: random.Random) -> None:
        # Vote (at least itself)
        candidates = list(communication_protocol.get_neighbors(only_direct=False))
        if state.addr not in candidates:
            candidates.append(state.addr)
        logger.debug(state.addr, f"👨‍🏫 {len(candidates)} candidates to train set")

        # Order candidates to make a deterministic vote (based on the random seed)
        candidates.sort()

        # Send vote 
        samples = min(state.experiment.trainset_size, len(candidates))
        nodes_voted = generator.sample(candidates, samples)
        weights = [math.floor(generator.randint(0, 1000) / (i + 1)) for i in range(samples)]
        votes = list(zip(nodes_voted, weights))

        # Adding votes
        state.train_set_votes[state.addr] = dict(votes)

        # Send and wait for votes
        logger.info(state.addr, "🗳️ Sending train set vote.")
        logger.debug(state.addr, f"🪞🗳️ Self Vote: {votes}")
        await communication_protocol.broadcast(
            communication_protocol.build_msg(
                VoteTrainSetCommand.get_name(),
                list(map(str, list(sum(votes, ())))),
                round=state.round,
            )
        )
