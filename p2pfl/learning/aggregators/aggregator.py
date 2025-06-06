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

"""Abstract aggregator."""

from typing import List

from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.utils.node_component import NodeComponent


class NoModelsToAggregateError(Exception):
    """Exception raised when there are no models to aggregate."""

    pass


class Aggregator(NodeComponent):
    """
    Class to manage the aggregation of models.

    Args:
        node_addr: Address of the node.

    """

    def __init__(self) -> None:
        """Initialize the aggregator."""
        self.partial_aggregation = False

        # (addr) Super
        NodeComponent.__init__(self)

    def aggregate(self, models: List[P2PFLModel]) -> P2PFLModel:
        """
        Aggregate the models.

        Args:
            models: Dictionary with the models to aggregate.

        """
        raise NotImplementedError

    def get_required_callbacks(self) -> List[str]:
        """
        Get the required callbacks for the aggregation.

        Returns:
            List of required callbacks.

        """
        return []
