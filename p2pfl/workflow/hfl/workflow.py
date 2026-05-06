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
    HFLEdgeLocalTrainStage,
    HFLEdgeSyncRootStage,
    HFLRootAggregateStage,
    HFLRootDistributeStage,
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

    Three-level hierarchy: workers train locally and send models to their
    assigned edge node. Edge nodes aggregate worker models and send the
    result to a root node. The root aggregates all edge models and
    distributes the global model back down through edges to workers.

    Flow (worker): setup -> worker_train -> worker_wait_global -> round_finished -> loop
    Flow (edge):   setup -> edge_local_train -> edge_aggregate_workers ->
                   edge_sync_root -> edge_distribute -> round_finished -> loop
    Flow (root):   setup -> root_aggregate -> root_distribute -> round_finished -> loop

    HFL-specific parameters are passed via ``experiment.data``:
        - ``role``: ``"worker"``, ``"edge"``, or ``"root"`` (required)
        - ``edge_addr``: edge address for workers (required for workers)
        - ``worker_addrs``: list of worker addresses (required for edges)
        - ``root_addr``: root address for edges (required for edges)
        - ``child_edge_addrs``: list of child edge addresses (required for root)
    """

    context_class = HFLContext

    def get_stages(self) -> list[Stage[HFLContext]]:
        """Return all HFL stages (worker, edge, and root paths)."""
        return [
            HFLSetupStage(),
            HFLWorkerTrainStage(),
            HFLWorkerWaitGlobalStage(),
            HFLEdgeLocalTrainStage(),
            HFLEdgeAggregateWorkersStage(),
            HFLEdgeSyncRootStage(),
            HFLEdgeDistributeStage(),
            HFLRootAggregateStage(),
            HFLRootDistributeStage(),
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
            root_addr=data.get("root_addr"),
            child_edge_addrs=data.get("child_edge_addrs", []),
            edge_trains=data.get("edge_trains", True),
        )

    def validate_experiment(self, ctx: HFLContext) -> None:
        """Validate HFL-specific experiment parameters."""
        if ctx.role not in ("worker", "edge", "root"):
            raise ValueError(f"Invalid HFL role: {ctx.role!r}. Must be 'worker', 'edge', or 'root'.")
        if ctx.role == "worker" and not ctx.edge_addr:
            raise ValueError("Workers must have an 'edge_addr' configured.")
        if ctx.role == "edge" and not ctx.worker_addrs:
            raise ValueError("Edges must have at least one worker in 'worker_addrs'.")
        if ctx.role == "edge" and not ctx.root_addr:
            raise ValueError("Edges must have a 'root_addr' configured.")
        if ctx.role == "root" and not ctx.child_edge_addrs:
            raise ValueError("Root must have at least one edge in 'child_edge_addrs'.")
