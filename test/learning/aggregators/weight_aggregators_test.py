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
"""
Weight aggregator tests - FedAvg, FedMedian, Krum, FedProx, FedOpt.

Tests verify expected aggregation behavior for each algorithm.
"""

import contextlib
import copy

import numpy as np
import pytest
from datasets import DatasetDict, load_dataset

from p2pfl.examples.mnist.model.mlp_pytorch import MLP
from p2pfl.learning.aggregators.aggregator import NoModelsToAggregateError
from p2pfl.learning.aggregators.fedavg import FedAvg
from p2pfl.learning.aggregators.fedmedian import FedMedian
from p2pfl.learning.aggregators.fedopt import FedAdagrad, FedAdam, FedYogi
from p2pfl.learning.aggregators.fedprox import FedProx
from p2pfl.learning.aggregators.krum import Krum
from p2pfl.learning.aggregators.pushsum import PushSum
from p2pfl.learning.aggregators.scaffold import Scaffold
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.learner_factory import LearnerFactory
from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel
from p2pfl.management.logger import logger
from p2pfl.workflow.engine.experiment import Experiment

from .mocks import WeightBasedModelMock

###
# PushSum Tests
###


def test_pushsum_weighted_aggregation():
    """PushSum: weighted sum using mixing_weight (no normalization), push_sum_weight propagation."""
    aggregator = PushSum()
    aggregator.set_address("test")

    # Model 1: mixing_weight=0.3, push_sum_weight=1.0, params=[1,2]
    # Model 2: mixing_weight=0.5, push_sum_weight=2.0, params=[3,4]
    # Eq. 5: accum = 0.3*[1,2] + 0.5*[3,4] = [0.3+1.5, 0.6+2.0] = [1.8, 2.6]
    # Eq. 6: mu = 0.3*1.0 + 0.5*2.0 = 0.3 + 1.0 = 1.3
    models = [
        WeightBasedModelMock(
            [np.array([1.0, 2.0])],
            num_samples=10,
            contributors=["n1"],
            additional_info={"mixing_weight": 0.3, "push_sum_weight": 1.0},
        ),
        WeightBasedModelMock(
            [np.array([3.0, 4.0])],
            num_samples=20,
            contributors=["n2"],
            additional_info={"mixing_weight": 0.5, "push_sum_weight": 2.0},
        ),
    ]
    result = aggregator.aggregate(models)

    assert np.allclose(result.get_parameters()[0], np.array([1.8, 2.6]))
    assert result.get_num_samples() == 30
    assert set(result.get_contributors()) == {"n1", "n2"}
    assert result.get_info("push_sum_weight") == pytest.approx(1.3)


def test_pushsum_defaults_to_unit_weights():
    """PushSum: missing info fields default to 1.0, behaving like plain sum."""
    aggregator = PushSum()
    aggregator.set_address("test")

    # No mixing_weight or push_sum_weight set -> defaults to 1.0
    # accum = 1.0*[2,4] + 1.0*[6,8] = [8,12]
    # mu = 1.0*1.0 + 1.0*1.0 = 2.0
    models = [
        WeightBasedModelMock([np.array([2.0, 4.0])], num_samples=5, contributors=["a"]),
        WeightBasedModelMock([np.array([6.0, 8.0])], num_samples=15, contributors=["b"]),
    ]
    result = aggregator.aggregate(models)

    assert np.allclose(result.get_parameters()[0], np.array([8.0, 12.0]))
    assert result.get_info("push_sum_weight") == pytest.approx(2.0)


def test_pushsum_single_model():
    """PushSum: single model with mixing_weight scales params correctly."""
    aggregator = PushSum()
    aggregator.set_address("test")

    models = [
        WeightBasedModelMock(
            [np.array([10.0, 20.0])],
            num_samples=42,
            contributors=["solo"],
            additional_info={"mixing_weight": 0.7, "push_sum_weight": 3.0},
        ),
    ]
    result = aggregator.aggregate(models)

    # accum = 0.7 * [10, 20] = [7, 14]
    # mu = 0.7 * 3.0 = 2.1
    assert np.allclose(result.get_parameters()[0], np.array([7.0, 14.0]))
    assert result.get_info("push_sum_weight") == pytest.approx(2.1)
    assert result.get_num_samples() == 42
    assert result.get_contributors() == ["solo"]


def test_pushsum_empty_models_raises():
    """PushSum: aggregating empty list raises NoModelsToAggregateError."""
    aggregator = PushSum()
    aggregator.set_address("test")

    with pytest.raises(NoModelsToAggregateError):
        aggregator.aggregate([])


def test_pushsum_multi_layer():
    """PushSum: correctly handles models with multiple parameter layers."""
    aggregator = PushSum()
    aggregator.set_address("test")

    models = [
        WeightBasedModelMock(
            [np.array([1.0]), np.array([10.0, 20.0])],
            num_samples=5,
            contributors=["n1"],
            additional_info={"mixing_weight": 0.4, "push_sum_weight": 1.0},
        ),
        WeightBasedModelMock(
            [np.array([3.0]), np.array([30.0, 40.0])],
            num_samples=5,
            contributors=["n2"],
            additional_info={"mixing_weight": 0.6, "push_sum_weight": 1.0},
        ),
    ]
    result = aggregator.aggregate(models)

    # Layer 0: 0.4*1 + 0.6*3 = 0.4 + 1.8 = 2.2
    # Layer 1: 0.4*[10,20] + 0.6*[30,40] = [4+18, 8+24] = [22, 32]
    assert np.allclose(result.get_parameters()[0], np.array([2.2]))
    assert np.allclose(result.get_parameters()[1], np.array([22.0, 32.0]))


###
# FedAvg Tests
###


def test_fedavg_aggregation():
    """FedAvg: weighted average by sample count."""
    aggregator = FedAvg()
    aggregator.set_address("test")

    # Weighted: [1,1]*10 + [2,2]*20 + [3,3]*30 = [140,140]/60 = [2.33, 2.33]
    models = [
        WeightBasedModelMock([np.array([1.0, 1.0])], num_samples=10, contributors=["n1"]),
        WeightBasedModelMock([np.array([2.0, 2.0])], num_samples=20, contributors=["n2"]),
        WeightBasedModelMock([np.array([3.0, 3.0])], num_samples=30, contributors=["n3"]),
    ]
    result = aggregator.aggregate(models)
    assert np.allclose(result.get_parameters()[0], np.array([140.0 / 60, 140.0 / 60]))
    assert result.get_num_samples() == 60
    assert set(result.get_contributors()) == {"n1", "n2", "n3"}

    # Equal samples: [0,0]*10 + [2,4]*10 = [1,2]
    models_equal = [
        WeightBasedModelMock([np.array([0.0, 0.0])], num_samples=10, contributors=["n1"]),
        WeightBasedModelMock([np.array([2.0, 4.0])], num_samples=10, contributors=["n2"]),
    ]
    result = aggregator.aggregate(models_equal)
    assert np.allclose(result.get_parameters()[0], np.array([1.0, 2.0]))


###
# FedMedian Tests
###


def test_fedmedian_aggregation():
    """FedMedian: computes median, ignores outliers, handles even count."""
    aggregator = FedMedian()
    aggregator.set_address("test")

    # Odd count: [1,1], [2,2], [3,3] → median = [2,2]
    models_odd = [
        WeightBasedModelMock([np.array([1.0, 1.0])], num_samples=10, contributors=["n1"]),
        WeightBasedModelMock([np.array([2.0, 2.0])], num_samples=10, contributors=["n2"]),
        WeightBasedModelMock([np.array([3.0, 3.0])], num_samples=10, contributors=["n3"]),
    ]
    result = aggregator.aggregate(models_odd)
    assert np.allclose(result.get_parameters()[0], np.array([2.0, 2.0]))
    assert result.get_num_samples() == 30

    # Byzantine resilience: [1,1], [1.1,1.1], [100,100] → median = [1.1,1.1]
    models_outlier = [
        WeightBasedModelMock([np.array([1.0, 1.0])], num_samples=10, contributors=["h1"]),
        WeightBasedModelMock([np.array([1.1, 1.1])], num_samples=10, contributors=["h2"]),
        WeightBasedModelMock([np.array([100.0, 100.0])], num_samples=10, contributors=["byz"]),
    ]
    result = aggregator.aggregate(models_outlier)
    assert np.allclose(result.get_parameters()[0], np.array([1.1, 1.1]))

    # Even count: [1], [2], [3], [4] → median = (2+3)/2 = 2.5
    models_even = [
        WeightBasedModelMock([np.array([1.0])], num_samples=10, contributors=["n1"]),
        WeightBasedModelMock([np.array([2.0])], num_samples=10, contributors=["n2"]),
        WeightBasedModelMock([np.array([3.0])], num_samples=10, contributors=["n3"]),
        WeightBasedModelMock([np.array([4.0])], num_samples=10, contributors=["n4"]),
    ]
    result = aggregator.aggregate(models_even)
    assert np.allclose(result.get_parameters()[0], np.array([2.5]))


###
# Krum Tests
###


def test_krum_selects_closest_to_majority():
    """Krum selects model with minimum total distance to others."""
    # 3 models: [0,0], [0.1,0.1], [100,100]
    # Distances:
    #   [0,0] to [0.1,0.1] ≈ 0.14, to [100,100] ≈ 141.4 → sum ≈ 141.5
    #   [0.1,0.1] to [0,0] ≈ 0.14, to [100,100] ≈ 141.3 → sum ≈ 141.4
    #   [100,100] to [0,0] ≈ 141.4, to [0.1,0.1] ≈ 141.3 → sum ≈ 282.7
    # Krum should select [0.1,0.1] (smallest sum)
    models = [
        WeightBasedModelMock([np.array([0.0, 0.0])], num_samples=10, contributors=["n1"]),
        WeightBasedModelMock([np.array([0.1, 0.1])], num_samples=10, contributors=["n2"]),
        WeightBasedModelMock([np.array([100.0, 100.0])], num_samples=10, contributors=["byzantine"]),
    ]

    aggregator = Krum()
    aggregator.set_address("test")
    result = aggregator.aggregate(models)

    # Should select [0.1, 0.1] or [0.0, 0.0] (both close to majority)
    params = result.get_parameters()[0]
    assert params[0] <= 0.1  # Either [0,0] or [0.1,0.1], not the outlier
    assert result.get_num_samples() == 30
    assert set(result.get_contributors()) == {"n1", "n2", "byzantine"}


###
# FedProx Tests
###


def test_fedprox_aggregation():
    """FedProx: weighted average, adds proximal_mu, requires callback."""
    # [1,1]*10 + [3,3]*20 = [70,70]/30 = [2.33, 2.33]
    models = [
        WeightBasedModelMock([np.array([1.0, 1.0])], num_samples=10, contributors=["n1"]),
        WeightBasedModelMock([np.array([3.0, 3.0])], num_samples=20, contributors=["n2"]),
    ]

    aggregator = FedProx(proximal_mu=0.1)
    aggregator.set_address("test")
    result = aggregator.aggregate(models)

    # Weighted average
    expected = np.array([70.0 / 30, 70.0 / 30])
    assert np.allclose(result.get_parameters()[0], expected)

    # Adds proximal_mu
    assert result.get_info("fedprox")["proximal_mu"] == 0.1

    # Requires callback
    assert "fedprox" in aggregator.get_required_callbacks()

    # Correct totals
    assert result.get_num_samples() == 30
    assert set(result.get_contributors()) == {"n1", "n2"}


@pytest.mark.asyncio
@pytest.mark.e2e_train
async def test_fedprox_e2e_two_rounds():
    """Test FedProx aggregator + callback integration over two rounds."""
    # Dataset
    dataset = P2PFLDataset(
        DatasetDict(
            {
                "train": load_dataset("p2pfl/MNIST", split="train[:100]"),
                "test": load_dataset("p2pfl/MNIST", split="test[:10]"),
            }
        )
    )

    # Create the model
    p2pfl_model = LightningModel(MLP())

    # Create FedProx aggregator
    aggregator = FedProx(proximal_mu=0.1)
    aggregator.set_address("test_fedprox")

    node_name = "fedprox-test-node"
    with contextlib.suppress(Exception):
        logger.register_node(node_name)
    experiment = Experiment(exp_name="test_fedprox", total_rounds=2)
    logger.experiment_started(node_name, experiment)

    # Learner with aggregator
    learner = LearnerFactory.create_learner(p2pfl_model)()
    learner.set_address(node_name)
    learner.set_model(p2pfl_model)
    learner.set_data(dataset)
    learner.indicate_aggregator(aggregator)  # This adds the FedProx callback

    # Round 1: First round - no proximal term (callback skips first round)
    learner.set_epochs(1)
    trained_model = await learner.fit()
    assert trained_model is not None
    assert trained_model.get_num_samples() == 100

    # Simulate aggregation: aggregate the model and set it back
    aggregated_model = aggregator.aggregate([trained_model])
    learner.set_model(aggregated_model)

    # Round 2: Proximal term applied (callback reads proximal_mu, snapshots params)
    trained_model_r2 = await learner.fit()
    assert trained_model_r2 is not None

    # Evaluate
    await learner.evaluate()


###
# FedOpt Tests (FedAdam, FedAdagrad, FedYogi)
###


@pytest.mark.parametrize("aggregator_cls", [FedAdam, FedAdagrad, FedYogi])
def test_fedopt_aggregation(aggregator_cls):
    """FedOpt: first round = FedAvg, state initialized, multi-round stable."""
    aggregator = aggregator_cls(eta=0.1)
    aggregator.set_address("test")

    # Round 1: [1,1]*10 + [3,3]*10 = [2,2]
    models_r1 = [
        WeightBasedModelMock([np.array([1.0, 1.0])], num_samples=10, contributors=["n1"]),
        WeightBasedModelMock([np.array([3.0, 3.0])], num_samples=10, contributors=["n2"]),
    ]
    result_r1 = aggregator.aggregate(models_r1)

    # First round equals weighted average (no prior state)
    assert np.allclose(result_r1.get_parameters()[0], np.array([2.0, 2.0]))

    # State initialized
    assert len(aggregator.current_weights) == 1
    assert aggregator.current_weights[0].shape == (2,)

    # Round 2-3: verify multi-round stability
    for i in range(2):
        models = [
            WeightBasedModelMock([np.array([2.0 + i, 2.0 + i])], num_samples=10, contributors=["n1"]),
        ]
        result = aggregator.aggregate(models)
        params = result.get_parameters()[0]

        # Output is valid (not NaN/Inf, reasonable range)
        assert not np.isnan(params).any()
        assert not np.isinf(params).any()
        assert np.all(np.abs(params) < 100)

    # Momentum state exists after multiple rounds
    assert len(aggregator.m_t) > 0
    assert len(aggregator.v_t) > 0


###
# Scaffold Tests
###


def test_scaffold_aggregator_requires_delta_info():
    """Scaffold: verify aggregator requires delta_y_i and delta_c_i from models."""
    # Model without scaffold info should fail aggregation
    model = LightningModel(MLP(), num_samples=24, contributors=["node1"])
    aggregator = Scaffold()
    aggregator.set_address("test")

    with pytest.raises(KeyError):
        aggregator.aggregate([model])


def test_scaffold_correct_aggregation():
    """Scaffold: verify correct mathematical aggregation of delta_y_i and delta_c_i."""
    aggr = Scaffold(global_lr=0.1)
    aggr.set_address("test")

    # Initial params
    initial_global_model_params = [np.array([0.0, 0.0]), np.array([0.0, 0.0])]
    aggr.global_model_params = copy.deepcopy(initial_global_model_params)
    aggr.c = [np.array([0.0, 0.0]), np.array([0.0, 0.0])]

    model1 = WeightBasedModelMock(
        params=[np.array([1.0, 1.0]), np.array([1.0, 1.0])],
        num_samples=10,
        additional_info={
            "scaffold": {
                "delta_y_i": [np.array([1.0, 1.0]), np.array([1.0, 1.0])],
                "delta_c_i": [np.array([1.0, 1.0]), np.array([1.0, 1.0])],
            }
        },
        contributors=["client1"],
    )

    model2 = WeightBasedModelMock(
        params=[np.array([2.0, 2.0]), np.array([2.0, 2.0])],
        num_samples=20,
        additional_info={
            "scaffold": {
                "delta_y_i": [np.array([2.0, 2.0]), np.array([2.0, 2.0])],
                "delta_c_i": [np.array([2.0, 2.0]), np.array([2.0, 2.0])],
            }
        },
        contributors=["client2"],
    )

    # Expected: weighted avg of delta_y_i scaled by global_lr
    total_samples = 10 + 20  # 30
    expected_delta_y = (1.0 * 10 + 2.0 * 20) / total_samples * aggr.global_lr  # 0.166...
    expected_params = [
        initial_global_model_params[0] + expected_delta_y,
        initial_global_model_params[1] + expected_delta_y,
    ]

    aggregated = aggr.aggregate([model1, model2])

    # Check params
    for aggr_param, expected_param in zip(aggregated.get_parameters(), expected_params, strict=False):
        assert np.allclose(aggr_param, expected_param, atol=1e-7)

    # Check contributors
    assert set(aggregated.get_contributors()) == {"client1", "client2"}

    # Check global control variates: mean of delta_c_i
    expected_c = (1.0 + 2.0) / 2  # 1.5
    for c_param in aggr.c:
        assert np.allclose(c_param, expected_c, atol=1e-7)
