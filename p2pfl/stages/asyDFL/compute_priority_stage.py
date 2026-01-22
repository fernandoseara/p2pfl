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
"""Wait aggregated models stage."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from p2pfl.stages.network_state.async_network_state import AsyncNetworkState

if TYPE_CHECKING:
    from p2pfl.node import Node


class ComputePriorityStage:
    """Compute priority utility for AsyncDFL workflow."""

    @staticmethod
    async def execute(network_state: AsyncNetworkState, node: Node) -> list[tuple[str, float]]:
        """
        Execute the stage. Perform neighbor selection and update models.

        Args:
            network_state: The async network state.
            node: The node to execute the stage on.

        """
        communication_protocol = node.get_communication_protocol()
        local_state = node.get_local_state()

        # Initialize a list to store neighbors with their computed priority
        neighbor_priorities = []

        # Calculate priority p(b_i,j) (Equation 40)
        local_loss = network_state.get_losses(node.address)
        avg_local_loss = sum(local_loss) / len(local_loss) if local_loss else 0.0  # Average loss
        for neighbor in list(communication_protocol.get_neighbors(only_direct=True)):
            neighbor_loss = network_state.get_losses(neighbor)
            avg_neighbor_loss = sum(neighbor_loss) / len(neighbor_loss) if neighbor_loss else 0.0

            priority = ComputePriorityStage.compute_priority(
                local_state.round or 0,
                network_state.get_push_time(neighbor) or 0,
                network_state.get_p2p_updating_idx(neighbor) or 0,
                network_state.get_round(neighbor) or 0,
                avg_local_loss,
                avg_neighbor_loss,
                5,
            )  # TODO: dmax parameter

            # Append the neighbor and their computed priority as a tuple
            neighbor_priorities.append((neighbor, priority))

        return neighbor_priorities

    @staticmethod
    def select_neighbors(priority: list) -> list:
        """
        Select the neighbors based on the priority p(b_i,j).

        Args:
            priority: The priority of selecting neighbors.

        Returns:
            list: The selected neighbors.

        """
        return sorted(priority, key=lambda x: x[1], reverse=True)[:3]  # Select top 3 neighbors based on priority

    @staticmethod
    def compute_priority(ti: int, tp_ij: int, tj: int, tl_ji: int, f_ti: float, f_tj: float, dmax: int):
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
        dij = abs((ti - tp_ij) - (tj - tl_ji)) / dmax
        loss_term = math.exp(abs(f_ti - f_tj)) / math.exp(1)  # Measures how different the loss (or objective) values are.
        priority = dij + (1 - dij) * loss_term

        return priority
