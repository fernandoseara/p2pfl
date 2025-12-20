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

"""Federated Averaging (FedAvg) Aggregator."""

from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError, compatible_with
from p2pfl.learning.frameworks import ModelType
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel

# Inspired by the implementation of flower. Thank you so much for taking FL to another level :)
#     Original implementation: https://flower.ai/blog/2024-02-14-federated-xgboost-with-flower/


@compatible_with(ModelType.BOOSTING_TREE)
class FedXgbCyclic(Aggregator):
    """
    Federated XGBoost Cyclic Aggregator.

    Implements cyclic training for XGBoost models in federated learning.
    In cyclic training, only one client participates per round and the model
    is passed sequentially between clients.

    Attributes:
        SUPPORTS_PARTIAL_AGGREGATION (bool): Whether partial aggregation is supported.

    """

    SUPPORTS_PARTIAL_AGGREGATION: bool = True

    def __init__(self, disable_partial_aggregation: bool = False) -> None:
        """
        Initialize the FedXgbCyclic aggregator.

        Args:
            disable_partial_aggregation (bool): Whether to disable partial aggregation
                (default is False).

        """
        super().__init__(disable_partial_aggregation=disable_partial_aggregation)

    def aggregate(self, models: list[P2PFLModel]) -> P2PFLModel:
        """
        Perform cyclic aggregation on XGBoost models.

        In cyclic training, only the model from the current client (last in the list)
        is selected and passed to the next round.

        Args:
            models (list[P2PFLModel]): List of models to aggregate.

        Returns:
            P2PFLModel: The selected model (last model in the list) as the aggregated result.

        Raises:
            NoModelsToAggregateError: If there are no models to aggregate.

        """
        if len(models) == 0:
            raise NoModelsToAggregateError(f"({self.addr}) Trying to aggregate models when there is no models")

        # En entrenamiento cíclico, solo se toma el modelo del cliente actual (el último de la lista)
        selected_model = models[-1]
        total_samples = selected_model.get_num_samples()
        contributors = selected_model.get_contributors()
        return selected_model.build_copy(params=selected_model.get_parameters(), num_samples=total_samples, contributors=contributors)
