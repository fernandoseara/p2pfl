#
# This file is part of the p2pfl distribution
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

"""Sequential Learning Aggregator."""

from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


class SequentialLearning(Aggregator):
    """
    Sequential Learning Aggregator - passes a single model through unchanged.

    In sequential learning (also known as cyclic learning), only one client
    participates per round and the model is passed sequentially between clients.
    This aggregator simply passes through the received model without modification.

    Use cases:
        - Cyclic federated learning where clients train one after another
        - Ring topologies where the model circulates through all nodes
        - Any scenario requiring pass-through aggregation without modification

    Unlike WeightAggregator or TreeAggregator, this aggregator accepts any model
    type (neural networks, tree ensembles, etc.) since it performs no actual
    aggregation - just model forwarding.

    Note:
        This aggregator expects exactly one model per aggregation round.
        Passing multiple models will raise a ValueError.

    Example:
        >>> aggregator = SequentialLearning()
        >>> aggregator.set_address("node1")
        >>> result = aggregator.aggregate([single_model])

    """

    SUPPORTS_PARTIAL_AGGREGATION: bool = True

    def __init__(self, disable_partial_aggregation: bool = False) -> None:
        """
        Initialize the SequentialLearning aggregator.

        Args:
            disable_partial_aggregation: Whether to disable partial aggregation.

        """
        super().__init__(disable_partial_aggregation=disable_partial_aggregation)

    def _accepts_model(self, model: P2PFLModel) -> bool:
        """Accept any model type."""
        return True

    def _aggregate(self, models: list[P2PFLModel]) -> P2PFLModel:
        """
        Pass through a single model unchanged.

        Args:
            models: List containing exactly one model.

        Returns:
            The model passed through as the aggregated result.

        Raises:
            NoModelsToAggregateError: If there are no models.
            ValueError: If more than one model is passed.

        """
        if len(models) == 0:
            raise NoModelsToAggregateError(f"({self.address}) No models to aggregate")

        if len(models) > 1:
            raise ValueError(f"({self.address}) SequentialLearning expects exactly one model, got {len(models)}")

        model = models[0]
        return model.build_copy(params=model.get_parameters(), num_samples=model.get_num_samples(), contributors=model.get_contributors())
