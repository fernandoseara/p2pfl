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

import json

import numpy as np

from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.learning.frameworks.xgboost.xgboost_model import XGBoostModel

# TODO: add datasets to Huggingface

# Inspired by the implementation of flower. Thank you so much for taking FL to another level :)
#     Original implementation:
#
# Hay que añadir robustez a la hora de usar un agregador, se propone meter un campo en el modelo type y que se compruebe aqui, hacemos un enum y un get para pillarlo de cada modelo y metemos un decorador en cada agregator.
#
# meter datasets en hf


class FedXgbBagging(Aggregator):
    """Paper: https://arxiv.org/abs/1602.05629."""

    SUPPORTS_PARTIAL_AGGREGATION: bool = True

    def __init__(self, disable_partial_aggregation: bool = False) -> None:
        """Initialize the aggregator."""
        super().__init__(disable_partial_aggregation=disable_partial_aggregation)

    def aggregate(self, models: list[P2PFLModel]) -> P2PFLModel:
        """
        Aggregate the models.

        Args:
            models: Dictionary with the models (node: model,num_samples).

        Returns:
            A P2PFLModel with the aggregated.

        Raises:
            NoModelsToAggregateError: If there are no models to aggregate.

        """
        # Check if there are models to aggregate
        if len(models) == 0:
            raise NoModelsToAggregateError(f"({self.addr}) Trying to aggregate models when there is no models")

        # Runtime type check: ensure all models are XGBoostModel instances
        for model in models:
            if not isinstance(model, XGBoostModel):
                raise TypeError(f"FedXgbBagging requires XGBoostModel instances, got {type(model)}")

        # Total Samples
        total_samples = sum([m.get_num_samples() for m in models])

        # Cast to XGBoostModel for type safety (already validated above)
        xgb_models: list[XGBoostModel] = [m for m in models if isinstance(m, XGBoostModel)]  # type: ignore[misc]

        # Get the first model as JSON bytes and parse
        params = xgb_models[0].get_parameters()
        if len(params) == 0:
            return models[0]

        model_bytes = params[0].tobytes()
        global_model_json = json.loads(model_bytes.decode("utf-8"))

        # Aggregate models from the rest of the clients
        if len(xgb_models) > 1:
            for m in xgb_models[1:]:
                model_bytes = m.get_parameters()[0].tobytes()
                current_model_json = json.loads(model_bytes.decode("utf-8"))
                global_model_json = aggregate_boosters(global_model_json, current_model_json)

        # Convert aggregated JSON back to bytes
        aggregated_bytes = json.dumps(global_model_json).encode("utf-8")
        model_np = np.frombuffer(aggregated_bytes, dtype=np.uint8)

        # Get contributors
        contributors: list[str] = []
        for m in xgb_models:
            contributors = contributors + m.get_contributors()

        # Return an aggregated p2pfl model
        returned_model = xgb_models[0].build_copy(params=[model_np], num_samples=total_samples, contributors=contributors)
        return returned_model


def _get_tree_nums(xgb_model: dict) -> tuple[int, int]:
    # Get the number of trees
    tree_num = int(xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"])
    # Get the number of parallel trees
    paral_tree_num = int(xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_parallel_tree"])
    return tree_num, paral_tree_num


def aggregate_boosters(
    bst_prev: dict | None,
    bst_curr: dict,
) -> dict:
    """Conduct bagging aggregation for given trees."""
    if not bst_prev:
        return bst_curr

    tree_num_prev, _ = _get_tree_nums(bst_prev)
    _, paral_tree_num_curr = _get_tree_nums(bst_curr)

    bst_prev["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"] = str(tree_num_prev + paral_tree_num_curr)
    iteration_indptr = bst_prev["learner"]["gradient_booster"]["model"]["iteration_indptr"]
    bst_prev["learner"]["gradient_booster"]["model"]["iteration_indptr"].append(iteration_indptr[-1] + paral_tree_num_curr)

    # Aggregate new trees
    trees_curr = bst_curr["learner"]["gradient_booster"]["model"]["trees"]
    for tree_count in range(paral_tree_num_curr):
        trees_curr[tree_count]["id"] = tree_num_prev + tree_count
        bst_prev["learner"]["gradient_booster"]["model"]["trees"].append(trees_curr[tree_count])
        bst_prev["learner"]["gradient_booster"]["model"]["tree_info"].append(0)

    return bst_prev
