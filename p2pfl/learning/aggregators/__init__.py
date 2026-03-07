"""
Aggregation algorithms for P2PFL.

Base classes:
    - ``Aggregator``: Abstract base class for all aggregators
    - ``WeightAggregator``: Base class for neural network aggregators
    - ``TreeAggregator``: Base class for tree ensemble aggregators

Standard aggregators:
    - ``FedAvg``: Federated Averaging
    - ``FedProx``: Federated Proximal
    - ``FedMedian``: Federated Median
    - ``Krum``: Byzantine-resilient aggregation
    - ``Scaffold``: SCAFFOLD algorithm
    - ``FedOptBase``, ``FedAdagrad``, ``FedAdam``, ``FedYogi``: Adaptive optimization
    - ``SequentialLearning``: Sequential model passing (any model type)

Tree-based aggregators:
    - ``FedXgbBagging``: XGBoost bagging aggregation

Utility functions:
    - ``get_default_aggregator``: Select default aggregator based on model type
"""

# Base classes
from p2pfl.learning.aggregators.aggregator import (
    Aggregator,
    IncompatibleModelError,
    NoModelsToAggregateError,
    TreeAggregator,
    WeightAggregator,
)

# Standard aggregators
from p2pfl.learning.aggregators.fedavg import FedAvg
from p2pfl.learning.aggregators.fedmedian import FedMedian

# FedOpt family
from p2pfl.learning.aggregators.fedopt import FedAdagrad, FedAdam, FedOptBase, FedYogi
from p2pfl.learning.aggregators.fedprox import FedProx

# Tree-based aggregators
from p2pfl.learning.aggregators.fedxgbbagging import FedXgbBagging
from p2pfl.learning.aggregators.krum import Krum

# Push-sum aggregator
from p2pfl.learning.aggregators.pushsum import PushSum
from p2pfl.learning.aggregators.scaffold import Scaffold

# Sequential learning
from p2pfl.learning.aggregators.sequential import SequentialLearning

# Type imports for default aggregator selection
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel, TreeBasedModel


def get_default_aggregator(model: P2PFLModel) -> "WeightAggregator | TreeAggregator":
    """
    Select the appropriate default aggregator based on model type.

    Args:
        model: The model to determine aggregator for.

    Returns:
        FedXgbBagging for tree-based models, FedAvg for weight-based models.

    """
    if isinstance(model, TreeBasedModel):
        return FedXgbBagging()
    return FedAvg()


__all__ = [
    # Base classes
    "Aggregator",
    "WeightAggregator",
    "TreeAggregator",
    "IncompatibleModelError",
    "NoModelsToAggregateError",
    # Standard aggregators
    "FedAvg",
    "FedProx",
    "FedMedian",
    "Krum",
    "Scaffold",
    "FedOptBase",
    "FedAdagrad",
    "FedAdam",
    "FedYogi",
    "SequentialLearning",
    # Push-sum aggregator
    "PushSum",
    # Tree-based aggregators
    "FedXgbBagging",
    # Utility functions
    "get_default_aggregator",
]
