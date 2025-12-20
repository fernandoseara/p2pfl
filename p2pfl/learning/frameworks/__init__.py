"""
Package for the learning frameworks in P2PFL.

This package contains all the integrations with the different learning frameworks.
"""

from enum import Enum

###
#   Framework Enum
###


class Framework(Enum):
    """Enum for the different learning frameworks."""

    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    FLAX = "flax"
    XGBOOST = "xgboost"


class ModelType(Enum):
    """
    Enum for the different model types.

    Used to ensure compatibility between frameworks and aggregators.
    """

    NEURAL_NETWORK = "neural_network"  # PyTorch, TensorFlow, Flax
    BOOSTING_TREE = "boosting_tree"  # XGBoost
