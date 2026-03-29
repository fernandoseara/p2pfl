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
"""Tree aggregator tests - FedXgbBagging."""

import pytest

xgb = pytest.importorskip("xgboost", reason="XGBoost not available")

from sklearn.datasets import make_classification  # noqa: E402

from p2pfl.learning.aggregators.fedxgbbagging import FedXgbBagging  # noqa: E402
from p2pfl.learning.frameworks.xgboost.xgboost_model import XGBoostModel  # noqa: E402


@pytest.fixture
def base_xgboost_model():
    """Single trained model to derive test models from."""
    X, y = make_classification(n_samples=50, n_features=10, n_classes=2, random_state=42)
    model = xgb.XGBClassifier(n_estimators=2, max_depth=2, random_state=42)
    model.fit(X, y)
    return XGBoostModel(model, num_samples=50, contributors=["base"])


def test_fedxgbbagging_merges_trees(base_xgboost_model):
    """FedXgbBagging merges trees from all models."""
    params = base_xgboost_model.get_parameters()
    initial_tree_count = len(params["learner"]["gradient_booster"]["model"]["trees"])

    model1 = base_xgboost_model.build_copy(params=params, num_samples=10, contributors=["node0"])
    model2 = base_xgboost_model.build_copy(params=params, num_samples=20, contributors=["node1"])
    model3 = base_xgboost_model.build_copy(params=params, num_samples=30, contributors=["node2"])

    aggregator = FedXgbBagging()
    aggregator.set_addr("test")
    result = aggregator.aggregate([model1, model2, model3])

    # Bagging combines all
    assert result.get_num_samples() == 60
    assert set(result.get_contributors()) == {"node0", "node1", "node2"}

    # Tree count increases (first model + parallel_tree from each additional)
    result_params = result.get_parameters()
    result_tree_count = len(result_params["learner"]["gradient_booster"]["model"]["trees"])
    assert result_tree_count > initial_tree_count
