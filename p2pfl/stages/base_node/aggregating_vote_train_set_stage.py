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

import random
from typing import TYPE_CHECKING

from p2pfl.management.logger import logger
from p2pfl.stages.stage import Stage

if TYPE_CHECKING:
    from p2pfl.node import Node

class AggregatingVoteTrainSetStage(Stage):
    """Vote Train Set Stage."""

    @staticmethod
    async def execute(
        node: Node,
        ) -> None:
        """Execute the stage."""
        # Aggregate votes
        node.get_local_state().train_set = AggregatingVoteTrainSetStage.__validate_train_set(
            await AggregatingVoteTrainSetStage.__aggregate_votes(node),
        )
        logger.info(
            node.address,
            f"🚂 Train set of {len(node.get_local_state().train_set)} nodes: {node.get_local_state().train_set}",
        )
        # Set Models To Aggregate
        #aggregator.set_nodes_to_aggregate(state.train_set)


    @staticmethod
    async def __aggregate_votes(node: Node) -> list[str]:
        logger.debug(node.address, "⏳ Waiting other node votes.")
        results: dict[str, int] = {}
        for node_vote in list(node.get_network_state().train_set_votes.values()):
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
        top = min(len(results_ordered), node.get_local_state().experiment.trainset_size)
        results_ordered = results_ordered[0:top]
        # Clear votes
        node.get_network_state().clear_all_votes()
        logger.info(node.address, f"🔢 Computed {len(node.train_set_votes)} votes.")
        return [i[0] for i in results_ordered]


    @staticmethod
    def __validate_train_set(train_set: list[str], node: Node) -> list[str]:
        # Verify if node set is valid
        # (can happend that a node was down when the votes were being processed)
        for tsn in train_set:
            if tsn not in list(node.get_communication_protocol().get_neighbors(only_direct=False)) and (tsn != node.address):
                train_set.remove(tsn)
        return train_set
