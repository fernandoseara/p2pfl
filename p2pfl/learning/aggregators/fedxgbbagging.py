#
# This file is part of the p2pfl distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2024 Pedro Guijas Bravo.
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

"""Federated XGBoost Bagging Aggregator."""

import copy

from p2pfl.learning.aggregators.aggregator import NoModelsToAggregateError, TreeAggregator
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel, TreeBasedModel

# Inspired by the implementation of flower. Thank you so much for taking FL to another level :)
#     Original implementation: https://flower.ai/blog/2023-11-29-federated-xgboost-with-bagging-aggregation/


class FedXgbBagging(TreeAggregator):
    """
    Federated XGBoost Bagging Aggregator.

    Inherits from ``TreeAggregator`` as this aggregator works with tree-based
    models (XGBoost) using bagging aggregation.

    Implements bagging-based aggregation for XGBoost models in federated learning.
    Trees from different clients are combined into a single ensemble.

    Attributes:
        SUPPORTS_PARTIAL_AGGREGATION (bool): Whether partial aggregation is supported.

    """

    SUPPORTS_PARTIAL_AGGREGATION: bool = True

    def __init__(self, disable_partial_aggregation: bool = False) -> None:
        """
        Initialize the FedXgbBagging aggregator.

        Args:
            disable_partial_aggregation (bool): Whether to disable partial aggregation
                (default is False).

        """
        super().__init__(disable_partial_aggregation=disable_partial_aggregation)

    def _aggregate(self, models: list[P2PFLModel]) -> P2PFLModel:
        """
        Aggregate the models using bagging.

        Args:
            models: List of models to aggregate.

        Returns:
            A P2PFLModel with the aggregated result.

        Raises:
            NoModelsToAggregateError: If there are no models to aggregate.

        """
        # Check if there are models to aggregate
        if len(models) == 0:
            raise NoModelsToAggregateError(f"({self.addr}) Trying to aggregate models when there is no models")

        # Total Samples
        total_samples = sum([m.get_num_samples() for m in models])

        # Cast to TreeBasedModel for type safety (already validated above)
        tree_models: list[TreeBasedModel] = [m for m in models if isinstance(m, TreeBasedModel)]

        # Get the first model's tree structure (now returns dict directly)
        global_model = tree_models[0].get_parameters()
        if len(global_model) == 0:
            return models[0]

        # Deep copy to avoid mutating the original
        global_model = copy.deepcopy(global_model)

        # Aggregate models from the rest of the clients
        if len(tree_models) > 1:
            for m in tree_models[1:]:
                current_model = m.get_parameters()
                global_model = FedXgbBagging._aggregate_boosters(global_model, current_model)

        # Get contributors
        contributors: list[str] = []
        for m in tree_models:
            contributors = contributors + m.get_contributors()

        # Return an aggregated p2pfl model with the dict params
        returned_model = tree_models[0].build_copy(params=global_model, num_samples=total_samples, contributors=contributors)
        return returned_model

    @staticmethod
    def _get_tree_nums(xgb_model: dict) -> tuple[int, int]:
        """
        Get the number of trees and parallel trees from an XGBoost model.

        Args:
            xgb_model: The XGBoost model in JSON dictionary format.

        Returns:
            A tuple containing (number of trees, number of parallel trees).

        """
        tree_num = int(xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"])
        paral_tree_num = int(xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_parallel_tree"])
        return tree_num, paral_tree_num

    @staticmethod
    def _aggregate_boosters(bst_prev: dict | None, bst_curr: dict) -> dict:
        """
        Conduct bagging aggregation for given XGBoost trees.

        Combines trees from the current booster into the previous booster's ensemble.

        Args:
            bst_prev: The previous aggregated booster in JSON format,
                or None if this is the first booster.
            bst_curr: The current booster to aggregate in JSON format.

        Returns:
            The aggregated booster containing trees from both inputs.

        """
        if not bst_prev:
            return bst_curr

        tree_num_prev, _ = FedXgbBagging._get_tree_nums(bst_prev)
        _, paral_tree_num_curr = FedXgbBagging._get_tree_nums(bst_curr)

        bst_prev["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"] = str(tree_num_prev + paral_tree_num_curr)
        iteration_indptr = bst_prev["learner"]["gradient_booster"]["model"]["iteration_indptr"]
        bst_prev["learner"]["gradient_booster"]["model"]["iteration_indptr"].append(iteration_indptr[-1] + paral_tree_num_curr)

        # Aggregate new trees (deep copy to avoid mutating the source model)
        trees_curr = copy.deepcopy(bst_curr["learner"]["gradient_booster"]["model"]["trees"])
        for tree_count in range(paral_tree_num_curr):
            trees_curr[tree_count]["id"] = tree_num_prev + tree_count
            bst_prev["learner"]["gradient_booster"]["model"]["trees"].append(trees_curr[tree_count])
            bst_prev["learner"]["gradient_booster"]["model"]["tree_info"].append(0)

        return bst_prev
