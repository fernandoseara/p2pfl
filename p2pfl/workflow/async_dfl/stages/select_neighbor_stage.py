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
"""Select neighbors stage for async DFL workflow."""

from p2pfl.workflow.engine.stage import Stage


class SelectNeighborsStage(Stage):
    """Select neighbors utility for AsyncDFL workflow."""

    @staticmethod
    async def execute(
        neighbor_priorities: list[tuple[str, float]],
    ) -> list[str]:
        """
        Execute the stage. Perform neighbor selection based on priorities.

        Args:
            neighbor_priorities: The list of neighbors with their priorities.

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
