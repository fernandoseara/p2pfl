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
"""Tree-based framework tests (XGBoost)."""

import os

import numpy as np
import pytest

# Try to import XGBoost - skip tests if not available
xgb = pytest.importorskip("xgboost", reason="XGBoost not available or missing OpenMP dependency")

from sklearn.datasets import make_classification, make_regression  # noqa: E402

from p2pfl.learning.aggregators.fedxgbbagging import FedXgbBagging  # noqa: E402
from p2pfl.learning.frameworks.xgboost.xgboost_model import XGBoostModel  # noqa: E402

####################################
#    XGBoost Serialization Tests
####################################


def test_xgboost_get_set_params_classifier():
    """Test setting and getting parameters with XGBClassifier."""
    # Create sample data
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)

    # Create and train a classifier
    model = xgb.XGBClassifier(n_estimators=5, max_depth=3, random_state=42)
    model.fit(X, y)

    # Wrap in P2PFLModel
    p2pfl_model = XGBoostModel(model)

    # Get parameters - now returns dict (parsed JSON structure)
    params = p2pfl_model.get_parameters()
    assert isinstance(params, dict)

    # Verify the tree structure is correct
    assert "learner" in params
    assert "gradient_booster" in params["learner"]
    model_data = params["learner"]["gradient_booster"]["model"]
    assert "trees" in model_data
    assert "tree_info" in model_data
    assert "gbtree_model_param" in model_data

    # Verify correct number of trees
    num_trees = int(model_data["gbtree_model_param"]["num_trees"])
    assert num_trees == 5  # n_estimators=5
    assert len(model_data["trees"]) == 5
    assert len(model_data["tree_info"]) == 5

    # Verify each tree has expected structure
    for tree in model_data["trees"]:
        assert "base_weights" in tree
        assert "left_children" in tree
        assert "right_children" in tree

    # Create a new model and set parameters
    new_model = xgb.XGBClassifier(n_estimators=5, max_depth=3, random_state=42)
    p2pfl_model2 = XGBoostModel(new_model)
    p2pfl_model2.set_parameters(params)

    # Verify predictions match
    pred1 = p2pfl_model.get_model().predict(X)
    pred2 = p2pfl_model2.get_model().predict(X)
    assert np.array_equal(pred1, pred2)


def test_xgboost_get_set_params_regressor():
    """Test setting and getting parameters with XGBRegressor."""
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)

    model = xgb.XGBRegressor(n_estimators=5, max_depth=3, random_state=42)
    model.fit(X, y)
    p2pfl_model = XGBoostModel(model)

    # Get/set round-trip
    params = p2pfl_model.get_parameters()
    new_model = xgb.XGBRegressor(n_estimators=5, max_depth=3, random_state=42)
    p2pfl_model2 = XGBoostModel(new_model)
    p2pfl_model2.set_parameters(params)

    # Verify predictions match
    pred1 = p2pfl_model.get_model().predict(X)
    pred2 = p2pfl_model2.get_model().predict(X)
    assert np.allclose(pred1, pred2)


def test_xgboost_encoding_decoding():
    """Test encoding and decoding preserves model state."""
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)

    model = xgb.XGBClassifier(n_estimators=5, max_depth=3, random_state=42)
    model.fit(X, y)
    p2pfl_model1 = XGBoostModel(model)

    # Encode -> decode -> set round-trip
    encoded = p2pfl_model1.encode_parameters()
    p2pfl_model2 = XGBoostModel(xgb.XGBClassifier())
    decoded_params, additional_info = p2pfl_model2.decode_parameters(encoded)
    p2pfl_model2.set_parameters(decoded_params)

    # Verify predictions match
    pred1 = p2pfl_model1.get_model().predict(X)
    pred2 = p2pfl_model2.get_model().predict(X)
    assert np.array_equal(pred1, pred2)


def test_xgboost_params_preserve_model_state():
    """Verify predictions match after serialization/deserialization."""
    # Create sample data
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)

    # Create and train a model
    model = xgb.XGBClassifier(n_estimators=10, max_depth=4, random_state=42)
    model.fit(X, y)

    # Wrap in P2PFLModel
    p2pfl_model1 = XGBoostModel(model)
    original_pred = p2pfl_model1.get_model().predict(X)

    # Serialize and deserialize
    params = p2pfl_model1.get_parameters()

    # Create new model and load params
    new_model = xgb.XGBClassifier()
    p2pfl_model2 = XGBoostModel(new_model)
    p2pfl_model2.set_parameters(params)

    # Verify predictions match
    new_pred = p2pfl_model2.get_model().predict(X)
    assert np.array_equal(original_pred, new_pred)


def test_xgboost_no_temp_files_created():
    """Ensure no temporary files are created during serialization."""
    # Create sample data
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)

    # Create and train a model
    model = xgb.XGBClassifier(n_estimators=5, max_depth=3, random_state=42)
    model.fit(X, y)

    # Wrap in P2PFLModel
    p2pfl_model = XGBoostModel(model)

    # Check that temp directory doesn't exist or is empty
    temp_dir = "temp_xgboost_json"

    # Get parameters multiple times
    for _ in range(3):
        params = p2pfl_model.get_parameters()
        # Set parameters
        new_model = xgb.XGBClassifier()
        p2pfl_model2 = XGBoostModel(new_model)
        p2pfl_model2.set_parameters(params)

    # Verify no temp files were created
    if os.path.exists(temp_dir):
        files = os.listdir(temp_dir)
        assert len(files) == 0, f"Temporary files found: {files}"


def test_xgboost_incremental_training():
    """Test incremental training with Booster object."""
    # Create sample data
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)

    # Train initial model
    model1 = xgb.XGBClassifier(n_estimators=5, max_depth=3, random_state=42)
    model1.fit(X, y)

    # Get initial accuracy
    initial_pred = model1.predict(X)
    initial_accuracy = np.mean(initial_pred == y)

    # Continue training (incremental)
    model2 = xgb.XGBClassifier(n_estimators=5, max_depth=3, random_state=42)
    booster = model1.get_booster()
    model2.fit(X, y, xgb_model=booster)

    # Get new accuracy (should be same or better)
    new_pred = model2.predict(X)
    new_accuracy = np.mean(new_pred == y)

    # Verify incremental training worked (accuracy should be at least as good)
    assert new_accuracy >= initial_accuracy


##############################
#    XGBoost Model Copy Tests
##############################


def test_xgboost_build_copy():
    """Test model copying with bytes."""
    # Create sample data
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)

    # Create and train a model
    model = xgb.XGBClassifier(n_estimators=5, max_depth=3, random_state=42)
    model.fit(X, y)

    # Wrap in P2PFLModel
    p2pfl_model = XGBoostModel(model, num_samples=100, contributors=["node1"])

    # Build a copy
    params = p2pfl_model.get_parameters()
    copied_model = p2pfl_model.build_copy(params=params, num_samples=200, contributors=["node2"])

    # Verify copy has correct metadata
    assert copied_model.get_num_samples() == 200
    assert copied_model.get_contributors() == ["node2"]

    # Verify predictions match original
    pred1 = p2pfl_model.get_model().predict(X)
    pred2 = copied_model.get_model().predict(X)
    assert np.array_equal(pred1, pred2)


##################################
#    XGBoost Metadata Tests
##################################


def test_xgboost_learner_fit():
    """Test XGBoostModel with metadata (num_samples, contributors)."""
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)

    model = xgb.XGBClassifier(n_estimators=5, max_depth=3, random_state=42)
    model.fit(X, y)
    p2pfl_model = XGBoostModel(model, num_samples=100, contributors=["node1"])

    # Verify metadata
    assert p2pfl_model.get_num_samples() == 100
    assert p2pfl_model.get_contributors() == ["node1"]

    # Verify round-trip preserves predictions
    params = p2pfl_model.get_parameters()
    p2pfl_model2 = XGBoostModel(xgb.XGBClassifier())
    p2pfl_model2.set_parameters(params)
    assert np.array_equal(p2pfl_model.get_model().predict(X), p2pfl_model2.get_model().predict(X))


##################################
#    XGBoost End-to-End Tests
##################################


def test_xgboost_e2e_dict_params_workflow():
    """End-to-end: train -> aggregate -> serialize -> deserialize -> predict."""
    X, y = make_classification(n_samples=200, n_features=20, n_classes=2, random_state=42)
    X_train, X_test = X[:150], X[150:]
    y_train = y[:150]

    # Train 3 models
    trained_models = []
    for i in range(3):
        model = xgb.XGBClassifier(n_estimators=2, max_depth=2, random_state=42 + i)
        model.fit(X_train, y_train)
        trained_models.append(XGBoostModel(model, num_samples=50, contributors=[f"node{i}"]))

    # Aggregate
    aggregator = FedXgbBagging()
    aggregator.set_address("e2e_test")
    aggregated_model = aggregator.aggregate(trained_models)

    # Verify tree count: 2 + 1 + 1 = 4 trees
    agg_params = aggregated_model.get_parameters()
    assert len(agg_params["learner"]["gradient_booster"]["model"]["trees"]) == 4

    # Serialize -> deserialize round-trip
    serialized = aggregated_model.encode_parameters()
    restored_model = XGBoostModel(xgb.XGBClassifier())
    restored_model.set_parameters(serialized)

    # Verify predictions match
    pred_agg = aggregated_model.get_model().predict(X_test)
    pred_restored = restored_model.get_model().predict(X_test)
    assert np.array_equal(pred_agg, pred_restored)


def test_xgboost_compression_with_dict_params():
    """Test that ByteCompressor works with dict params (TensorCompressors skipped)."""
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)

    model = xgb.XGBClassifier(n_estimators=3, max_depth=2, random_state=42)
    model.fit(X, y)

    # Test with zlib compression (ByteCompressor - should work with dict params)
    p2pfl_model = XGBoostModel(model, compression={"zlib": {}})
    encoded = p2pfl_model.encode_parameters()

    # Decode and verify predictions match
    restored = XGBoostModel(xgb.XGBClassifier())
    restored.set_parameters(encoded)
    assert np.array_equal(p2pfl_model.get_model().predict(X), restored.get_model().predict(X))
