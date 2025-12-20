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

"""XGBoost model wrapper for P2PFL."""

from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.exceptions import NotFittedError

from p2pfl.learning.frameworks import Framework, ModelType
from p2pfl.learning.frameworks.exceptions import ModelNotMatchingError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


class XGBoostModel(P2PFLModel):
    """
    P2PFL model abstraction for XGBoost.

    Wraps an XGBoost Booster or XGBClassifier for federated updates,
    providing serialization and deserialization of model parameters.

    Args:
        model (xgb.XGBModel): The XGBoost model to encapsulate.
        params (list[np.ndarray] | bytes | None): Serialized parameters
            (default is None).
        num_samples (int | None): Number of samples used in training
            (default is None).
        contributors (list[str] | None): List of contributor IDs
            (default is None).
        additional_info (dict[str, Any] | None): Extra metadata
            (default is None).
        compression (dict[str, dict[str, Any]] | None): Optional compression settings
            (default is None).
        id (int | None): Model identifier (default is None).

    Attributes:
        id (int): The model identifier.

    Raises:
        ModelNotMatchingError: If the provided model is not an XGBoost sklearn model.

    """

    def __init__(
        self,
        model: xgb.XGBModel,
        params: list[np.ndarray] | bytes | None = None,
        num_samples: int | None = None,
        contributors: list[str] | None = None,
        additional_info: dict[str, Any] | None = None,
        compression: dict[str, dict[str, Any]] | None = None,
        id: int | None = None,
    ) -> None:
        """
        Initialize the XGBoost model wrapper.

        Args:
            model (xgb.XGBModel): The XGBoost model to encapsulate.
            params (list[np.ndarray] | bytes | None): Serialized parameters
                (default is None).
            num_samples (int | None): Number of samples used in training
                (default is None).
            contributors (list[str] | None): List of contributor IDs
                (default is None).
            additional_info (dict[str, Any] | None): Extra metadata
                (default is None).
            compression (dict[str, dict[str, Any]] | None): Optional compression settings
                (default is None).
            id (int | None): Model identifier (default is None).

        Raises:
            ModelNotMatchingError: If the provided model is not an XGBoost sklearn model.

        """
        if not isinstance(model, xgb.XGBModel):
            raise ModelNotMatchingError("Provided model is not an XGBoost sklearn model")
        super().__init__(model, params, num_samples, contributors, additional_info, compression)
        self.id = id if id is not None else 0  # Default ID if not provided

    def get_parameters(self) -> list[np.ndarray]:
        """
        Extract model parameters as numpy arrays.

        Returns:
            list[np.ndarray]: List containing a single numpy array with serialized
                model bytes, or empty list if model is not fitted.

        """
        try:
            # Get underlying booster and serialize to JSON bytes
            booster = self.model.get_booster()
            model_bytes = booster.save_raw(raw_format="json")
            # Convert to numpy array for compatibility with P2PFL
            model_np = np.frombuffer(model_bytes, dtype=np.uint8)
            return [model_np]
        except NotFittedError:
            return []

    def set_parameters(self, params: list[np.ndarray] | bytes) -> None:
        """
        Set model parameters from numpy arrays or serialized bytes.

        Args:
            params (list[np.ndarray] | bytes): List of ndarrays or serialized bytes
                containing the model parameters.

        Raises:
            ModelNotMatchingError: If loading fails.
            TypeError: If params is not a list after decoding.

        """
        # If bytes, decode compression first
        if isinstance(params, bytes):
            params, _ = self.decode_parameters(params)

        if params is None or len(params) == 0:
            return

        # Type guard: at this point params should be list[np.ndarray]
        if not isinstance(params, list):
            raise TypeError(f"Expected params to be a list, got {type(params)}")

        # Convert numpy array back to bytearray
        model_bytes = bytearray(params[0].tobytes())

        # Load into the existing model (no need to create new instance or detect type)
        self.model.load_model(model_bytes)

    def get_framework(self) -> str:
        """
        Return the framework name identifier.

        Returns:
            str: The framework name ('xgboost').

        """
        return Framework.XGBOOST.value

    def get_model_type(self) -> str:
        """
        Retrieve the model type for aggregator compatibility.

        Returns:
            str: The model type identifier for boosting trees.

        """
        return ModelType.BOOSTING_TREE.value
