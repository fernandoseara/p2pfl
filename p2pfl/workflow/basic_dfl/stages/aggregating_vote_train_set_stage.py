#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
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
"""Vote Train Set Stage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from p2pfl.management.logger import logger
from p2pfl.workflow.engine.stage import Stage

if TYPE_CHECKING:
    from p2pfl.node import Node
    from p2pfl.workflow.basic_dfl.workflow import BasicPeerState


class AggregatingVoteTrainSetStage(Stage):
    """Vote Train Set Stage."""

    @staticmethod
    async def execute(
        peers: dict[str, BasicPeerState],
        node: Node,
    ) -> None:
        """Execute the stage."""
        from p2pfl.workflow.basic_dfl.workflow import BasicDFL

        workflow = node.workflow
        if not isinstance(workflow, BasicDFL):
            raise TypeError("Expected BasicDFL workflow for this stage")

        # Aggregate votes
        workflow.train_set = AggregatingVoteTrainSetStage.__validate_train_set(
            await AggregatingVoteTrainSetStage.__aggregate_votes(peers, node),
            node,
        )
        logger.info(
            node.address,
            f"🚂 Train set of {len(workflow.train_set)} nodes: {workflow.train_set}",
        )

    @staticmethod
    async def __aggregate_votes(peers: dict[str, BasicPeerState], node: Node) -> list[str]:
        assert node.workflow is not None
        # Get all votes
        results: dict[str, int] = {}
        for peer in peers.values():
            for k, v in peer.votes.items():
                if k in results:
                    results[k] += v
                else:
                    results[k] = v

        # Order by votes and get TOP X
        results_ordered = sorted(results.items(), key=lambda x: x[0])  # alphabetical tie-breaking for equal votes
        results_ordered = sorted(results_ordered, key=lambda x: x[1], reverse=True)
        experiment = node.workflow.experiment
        if experiment is None or experiment.trainset_size is None:
            raise ValueError("Experiment or trainset_size not initialized")
        top = min(len(results_ordered), experiment.trainset_size)
        results_ordered = results_ordered[0:top]
        # Clear votes
        num_votes = sum(len(p.votes) for p in peers.values())
        for p in peers.values():
            p.votes.clear()
        logger.info(node.address, f"🔢 Computed {num_votes} votes.")
        return [i[0] for i in results_ordered]

    @staticmethod
    def __validate_train_set(train_set: list[str], node: Node) -> list[str]:
        # Verify if node set is valid
        # (can happen that a node was down when the votes were being processed)
        neighbors = list(node.communication_protocol.get_neighbors(only_direct=False))
        return [tsn for tsn in train_set if tsn in neighbors or tsn == node.address]
