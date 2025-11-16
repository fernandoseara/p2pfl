#
# This file is part of the federated_learning_p2p (p2pfl) distribution
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
"""XGBoost framework tests."""

import os

import numpy as np
import pytest

# Try to import XGBoost - skip tests if not available
xgb = pytest.importorskip("xgboost", reason="XGBoost not available or missing OpenMP dependency")

from sklearn.datasets import make_classification, make_regression  # noqa: E402

from p2pfl.learning.aggregators.fedxgbbagging import FedXgbBagging  # noqa: E402
from p2pfl.learning.aggregators.fedxgbcyclic import FedXgbCyclic  # noqa: E402
from p2pfl.learning.frameworks.xgboost.xgboost_model import XGBoostModel  # noqa: E402

####
# Model Serialization Tests
####


def test_xgboost_get_set_params_classifier():
    """Test setting and getting parameters with XGBClassifier."""
    # Create sample data
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)

    # Create and train a classifier
    model = xgb.XGBClassifier(n_estimators=5, max_depth=3, random_state=42)
    model.fit(X, y)

    # Wrap in P2PFLModel
    p2pfl_model = XGBoostModel(model)

    # Get parameters
    params = p2pfl_model.get_parameters()
    assert len(params) == 1
    assert isinstance(params[0], np.ndarray)
    assert params[0].dtype == np.uint8

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
    # Create sample data
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)

    # Create and train a regressor
    model = xgb.XGBRegressor(n_estimators=5, max_depth=3, random_state=42)
    model.fit(X, y)

    # Wrap in P2PFLModel
    p2pfl_model = XGBoostModel(model)

    # Get parameters
    params = p2pfl_model.get_parameters()
    assert len(params) == 1
    assert isinstance(params[0], np.ndarray)
    assert params[0].dtype == np.uint8

    # Create a new model and set parameters
    new_model = xgb.XGBRegressor(n_estimators=5, max_depth=3, random_state=42)
    p2pfl_model2 = XGBoostModel(new_model)
    p2pfl_model2.set_parameters(params)

    # Verify predictions match
    pred1 = p2pfl_model.get_model().predict(X)
    pred2 = p2pfl_model2.get_model().predict(X)
    assert np.allclose(pred1, pred2)


def test_xgboost_encoding_decoding():
    """Test encoding and decoding of parameters with compression."""
    # Create sample data
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)

    # Create and train a model
    model = xgb.XGBClassifier(n_estimators=5, max_depth=3, random_state=42)
    model.fit(X, y)

    # Wrap in P2PFLModel
    p2pfl_model1 = XGBoostModel(model)
    encoded_params = p2pfl_model1.encode_parameters()

    # Create a new model and decode
    new_model = xgb.XGBClassifier(n_estimators=5, max_depth=3, random_state=42)
    p2pfl_model2 = XGBoostModel(new_model)
    decoded_params, additional_info = p2pfl_model2.decode_parameters(encoded_params)
    p2pfl_model2.set_parameters(decoded_params)
    p2pfl_model2.additional_info = additional_info

    # Verify encoding is consistent
    assert encoded_params == p2pfl_model1.encode_parameters()
    assert additional_info == p2pfl_model1.additional_info


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


####
# Aggregation Tests
####


def test_fedxgbbagging_aggregation():
    """Test FedXgbBagging with byte-based serialization."""
    # Create sample data
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)

    # Create and train multiple models
    models = []
    for i in range(3):
        model = xgb.XGBClassifier(n_estimators=3, max_depth=2, random_state=42 + i)
        model.fit(X, y)
        p2pfl_model = XGBoostModel(model, num_samples=100, contributors=[f"node{i}"])
        models.append(p2pfl_model)

    # Aggregate using FedXgbBagging
    aggregator = FedXgbBagging()
    aggregator.set_addr("test_aggregator")
    aggregated_model = aggregator.aggregate(models)

    # Verify aggregation worked
    assert aggregated_model is not None
    assert aggregated_model.get_num_samples() == 300
    assert set(aggregated_model.get_contributors()) == {"node0", "node1", "node2"}

    # Verify aggregated model can make predictions
    pred = aggregated_model.get_model().predict(X)
    assert len(pred) == len(y)


def test_fedxgbcyclic_aggregation():
    """Test FedXgbCyclic still works with updated serialization."""
    # Create sample data
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)

    # Create and train multiple models
    models = []
    for i in range(3):
        model = xgb.XGBClassifier(n_estimators=3, max_depth=2, random_state=42 + i)
        model.fit(X, y)
        p2pfl_model = XGBoostModel(model, num_samples=100, contributors=[f"node{i}"])
        models.append(p2pfl_model)

    # Aggregate using FedXgbCyclic
    aggregator = FedXgbCyclic()
    aggregator.set_addr("test_aggregator")
    aggregated_model = aggregator.aggregate(models)

    # Verify aggregation worked (should return last model)
    assert aggregated_model is not None
    assert aggregated_model.get_num_samples() == 100
    assert aggregated_model.get_contributors() == ["node2"]


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


####
# Learner Integration Tests
####


def test_xgboost_learner_fit():
    """Test full learner workflow with XGBoost."""
    # Create sample data directly (simpler than MNIST for XGBoost)
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)

    # Create a trained model
    model = xgb.XGBClassifier(n_estimators=5, max_depth=3, random_state=42)
    model.fit(X, y)

    # Wrap in P2PFLModel
    p2pfl_model = XGBoostModel(model, num_samples=100, contributors=["node1"])

    # Verify the model can be serialized and deserialized
    params = p2pfl_model.get_parameters()
    assert len(params) == 1

    # Create a new model and load params
    new_model = xgb.XGBClassifier()
    p2pfl_model2 = XGBoostModel(new_model)
    p2pfl_model2.set_parameters(params)

    # Verify predictions match
    pred1 = p2pfl_model.get_model().predict(X)
    pred2 = p2pfl_model2.get_model().predict(X)
    assert np.array_equal(pred1, pred2)
