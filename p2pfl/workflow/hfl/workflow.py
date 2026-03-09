#
# This file is part of the p2pfl (see https://github.com/pguijas/p2pfl).
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
"""HFL: Hierarchical Federated Learning workflow."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from p2pfl.workflow.engine.experiment import Experiment
from p2pfl.workflow.engine.stage import Stage
from p2pfl.workflow.engine.workflow import Workflow
from p2pfl.workflow.hfl.context import HFLContext
from p2pfl.workflow.hfl.stages import (
    HFLEdgeAggregateWorkersStage,
    HFLEdgeDistributeStage,
    HFLEdgeGossipStage,
    HFLEdgeLocalTrainStage,
    HFLRoundFinishedStage,
    HFLSetupStage,
    HFLWorkerTrainStage,
    HFLWorkerWaitGlobalStage,
)

if TYPE_CHECKING:
    from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
    from p2pfl.learning.aggregators.aggregator import Aggregator
    from p2pfl.learning.frameworks.learner import Learner


class HFL(Workflow[HFLContext]):
    """
    Hierarchical Federated Learning workflow.

    Two-level hierarchy: workers train locally and send models to their
    assigned edge node. Edge nodes aggregate worker models, perform
    inter-edge gossip for global consensus, then distribute the global
    model back to workers.

    Flow (worker): setup -> worker_train -> worker_wait_global -> round_finished -> loop
    Flow (edge):   setup -> edge_local_train -> edge_aggregate_workers ->
                   edge_gossip -> edge_distribute -> round_finished -> loop

    HFL-specific parameters are passed via ``experiment.data``:
        - ``role``: ``"worker"`` or ``"edge"`` (required)
        - ``edge_addr``: edge address for workers (required for workers)
        - ``worker_addrs``: list of worker addresses (required for edges)
        - ``edge_peers``: list of other edge addresses (optional for edges)
    """

    context_class = HFLContext

    def get_stages(self) -> list[Stage[HFLContext]]:
        """Return all HFL stages (both worker and edge paths)."""
        return [
            HFLSetupStage(),
            HFLWorkerTrainStage(),
            HFLWorkerWaitGlobalStage(),
            HFLEdgeLocalTrainStage(),
            HFLEdgeAggregateWorkersStage(),
            HFLEdgeGossipStage(),
            HFLEdgeDistributeStage(),
            HFLRoundFinishedStage(),
        ]

    def create_context(
        self,
        address: str,
        learner: Learner,
        aggregator: Aggregator,
        cp: CommunicationProtocol,
        generator: random.Random,
        experiment: Experiment,
    ) -> HFLContext:
        """Build HFL context, extracting role and topology from experiment.data."""
        data = experiment.data
        role = data.get("role", "worker")
        return HFLContext(
            address=address,
            learner=learner,
            aggregator=aggregator,
            cp=cp,
            generator=generator,
            experiment=experiment,
            role=role,
            edge_addr=data.get("edge_addr"),
            worker_addrs=data.get("worker_addrs", []),
            edge_peers=data.get("edge_peers", []),
        )

    def validate_experiment(self, ctx: HFLContext) -> None:
        """Validate HFL-specific experiment parameters."""
        if ctx.role not in ("worker", "edge"):
            raise ValueError(f"Invalid HFL role: {ctx.role!r}. Must be 'worker' or 'edge'.")
        if ctx.role == "worker" and not ctx.edge_addr:
            raise ValueError("Workers must have an 'edge_addr' configured.")
        if ctx.role == "edge" and not ctx.worker_addrs:
            raise ValueError("Edges must have at least one worker in 'worker_addrs'.")
