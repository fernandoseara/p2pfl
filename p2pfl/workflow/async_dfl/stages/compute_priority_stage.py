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
"""Compute priority stage for async DFL workflow."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from p2pfl.workflow.engine.stage import Stage

if TYPE_CHECKING:
    from p2pfl.node import Node
    from p2pfl.workflow.async_dfl.workflow import AsyncPeerState


class ComputePriorityStage(Stage):
    """Compute priority utility for AsyncDFL workflow."""

    @staticmethod
    async def execute(peers: dict[str, AsyncPeerState], node: Node) -> list[tuple[str, float]]:
        """
        Execute the stage. Perform neighbor selection and update models.

        Args:
            peers: The per-peer state dict.
            node: The node to execute the stage on.

        """
        assert node.workflow is not None
        communication_protocol = node.communication_protocol
        workflow = node.workflow

        # Initialize a list to store neighbors with their computed priority
        neighbor_priorities = []

        # Calculate priority p(b_i,j) (Equation 40)
        local_peer = peers.get(node.address)
        local_losses = local_peer.losses if local_peer else []
        avg_local_loss = sum(local_losses) / len(local_losses) if local_losses else 0.0
        for neighbor in list(communication_protocol.get_neighbors(only_direct=True)):
            neighbor_peer = peers.get(neighbor)
            if neighbor_peer is None:
                continue
            avg_neighbor_loss = sum(neighbor_peer.losses) / len(neighbor_peer.losses) if neighbor_peer.losses else 0.0

            priority = ComputePriorityStage.compute_priority(
                workflow.round,
                neighbor_peer.push_time,
                neighbor_peer.p2p_updating_idx,
                neighbor_peer.round_number,
                avg_local_loss,
                avg_neighbor_loss,
                5,
            )  # TODO: dmax parameter

            # Append the neighbor and their computed priority as a tuple
            neighbor_priorities.append((neighbor, priority))

        return neighbor_priorities

    @staticmethod
    def compute_priority(ti: int, tp_ij: int, tj: int, tl_ji: int, f_ti: float, f_tj: float, dmax: int) -> float:
        """
        Compute the priority p(b_ij) based on the given formula.

        Args:
            ti: Index of the local iteration on node i.
            tp_ij: Index of the local iteration when node i pushes model to node j.
            tj: Index of the local iteration on node j.
            tl_ji: Index of the local iteration when node j updates with model from node i.
            f_ti: Training loss or function value at node i.
            f_tj: Training loss or function value at node j.
            dmax: Maximum communication frequency bound.

        Returns:
            float: The computed priority p(b_ij).

        """
        if dmax <= 0:
            raise ValueError("dmax must be positive")

        # Interprets the synchronization lag between nodes.
        # Expected values: 0 (perfect sync) to 1 (max staleness).
        dij = min(abs((ti - tp_ij) - (tj - tl_ji)) / dmax, 1.0)
        # Measures how different the loss (or objective) values are.
        try:
            loss_term = math.exp(abs(f_ti - f_tj)) / math.exp(1)
        except OverflowError:
            loss_term = float("inf")
        priority = dij + (1 - dij) * loss_term

        return priority
