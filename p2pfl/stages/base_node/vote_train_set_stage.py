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

            # Aggregate votes
            state.train_set = VoteTrainSetStage.__validate_train_set(
                await VoteTrainSetStage.__aggregate_votes(state, communication_protocol),
                state,
                communication_protocol,
            )
            logger.info(
                state.addr,
                f"🚂 Train set of {len(state.train_set)} nodes: {state.train_set}",
            )

            # Set Models To Aggregate
            aggregator.set_nodes_to_aggregate(state.train_set)
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
        state.train_set_votes_lock.acquire()
        state.train_set_votes[state.addr] = dict(votes)
        state.train_set_votes_lock.release()

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

    @staticmethod
    async def __aggregate_votes(state: NodeState, communication_protocol: CommunicationProtocol) -> list[str]:
        logger.debug(state.addr, "⏳ Waiting other node votes.")

        # Get time
        count = 0.0
        begin = time.time()

        while True:
            # If the trainning has been interrupted, stop waiting
            check_early_stop(state)

            # Update time counters (timeout)
            count = count + (time.time() - begin)
            begin = time.time()
            timeout = count > Settings.training.VOTE_TIMEOUT

            # Clear non candidate votes
            state.train_set_votes_lock.acquire()
            nc_votes = {
                k: v
                for k, v in state.train_set_votes.items()
                if k in list(communication_protocol.get_neighbors(only_direct=False)) or k == state.addr
            }
            state.train_set_votes_lock.release()

            # Determine if all votes are received
            needed_votes = set(list(communication_protocol.get_neighbors(only_direct=False)) + [state.addr])
            votes_ready = needed_votes == set(nc_votes.keys())

            if votes_ready or timeout:
                if timeout and not votes_ready:
                    missing_votes = set(list(communication_protocol.get_neighbors(only_direct=False)) + [state.addr]) - set(nc_votes.keys())
                    logger.info(
                        state.addr,
                        f"Timeout for vote aggregation. Missing votes from {missing_votes}",
                    )

                results: dict[str, int] = {}
                for node_vote in list(nc_votes.values()):
                    for i in range(len(node_vote)):
                        k = list(node_vote.keys())[i]
                        v = list(node_vote.values())[i]
                        if k in results:
                            results[k] += v
                        else:
                            results[k] = v

                # Order by votes and get TOP X
                results_ordered = sorted(
                    results.items(), key=lambda x: x[0], reverse=True
                )  # to equal solve of draw (node name alphabetical order)
                results_ordered = sorted(results_ordered, key=lambda x: x[1], reverse=True)
                top = min(len(results_ordered), state.experiment.trainset_size)
                results_ordered = results_ordered[0:top]

                # Clear votes
                state.train_set_votes = {}
                logger.info(state.addr, f"🔢 Computed {len(nc_votes)} votes.")
                return [i[0] for i in results_ordered]

            # Wait for votes or refresh every 2 seconds
            state.wait_votes_ready_lock.acquire(timeout=2)

    @staticmethod
    def __validate_train_set(
        train_set: list[str],
        state: NodeState,
        communication_protocol: CommunicationProtocol,
    ) -> list[str]:
        # Verify if node set is valid
        # (can happend that a node was down when the votes were being processed)
        for tsn in train_set:
            if tsn not in list(communication_protocol.get_neighbors(only_direct=False)) and (tsn != state.addr):
                train_set.remove(tsn)
        return train_set
