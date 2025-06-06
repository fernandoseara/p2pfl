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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from p2pfl.node import Node

class SelectNeighborsStage:
    """Compute priority stage."""

    @staticmethod
    async def execute(
        neighbor_priorities: list[tuple[str, float]],
        node: Node,
        ) -> list[str]:
        """
        Execute the stage. Perform neighbor selection and update models.

        Args:
            node: The node to execute the stage on.
        """
        # Sort the neighbors by their priority in descending order (highest priority first)
        neighbor_priorities.sort(key=lambda x: x[1], reverse=True)

        # Construct selected neighbor nodes set N*_i,t
        selected_neighbors: list[str] = []
        # Greedily select neighbors based on highest priority
        for neighbor, _ in neighbor_priorities:
            # Apply a greedy selection strategy

            # TODO: adjust the condition
            if len(selected_neighbors) < 3:
                selected_neighbors.append(neighbor)
            else:
                break  # Stop once you reach the limit

        return selected_neighbors

    @staticmethod
    def select_neighbors(priority: list) -> list:
        """
        Select the neighbors based on the priority p(b_i,j) of node i selecting neighbor j.

        Parameters:
            priority: The priority of selecting neighbors.

        Returns:
            list: The selected neighbors.

        """
        pass
