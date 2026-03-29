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

"""XGBoost model wrapper for P2PFL."""

import json
from typing import Any

import xgboost as xgb
from sklearn.exceptions import NotFittedError

from p2pfl.learning.frameworks import Framework
from p2pfl.learning.frameworks.exceptions import ModelNotMatchingError
from p2pfl.learning.frameworks.p2pfl_model import TreeBasedModel


class XGBoostModel(TreeBasedModel):
    """
    P2PFL model abstraction for XGBoost.

    Wraps an XGBoost Booster or XGBClassifier for federated updates,
    providing serialization and deserialization of model parameters.

    Args:
        model (xgb.XGBModel): The XGBoost model to encapsulate.
        params (dict[str, Any] | bytes | None): Tree structure dict or serialized bytes
            (default is None).
        num_samples (int | None): Number of samples used in training
            (default is None).
        contributors (list[str] | None): List of contributor IDs
            (default is None).
        additional_info (dict[str, Any] | None): Extra metadata
            (default is None).
        compression (dict[str, dict[str, Any]] | None): Optional compression settings
            (default is None).

    Raises:
        ModelNotMatchingError: If the provided model is not an XGBoost sklearn model.

    """

    def __init__(
        self,
        model: xgb.XGBModel,
        params: dict[str, Any] | bytes | None = None,
        num_samples: int | None = None,
        contributors: list[str] | None = None,
        additional_info: dict[str, Any] | None = None,
        compression: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """
        Initialize the XGBoost model wrapper.

        Args:
            model (xgb.XGBModel): The XGBoost model to encapsulate.
            params (dict[str, Any] | bytes | None): Tree structure dict or serialized bytes
                (default is None).
            num_samples (int | None): Number of samples used in training
                (default is None).
            contributors (list[str] | None): List of contributor IDs
                (default is None).
            additional_info (dict[str, Any] | None): Extra metadata
                (default is None).
            compression (dict[str, dict[str, Any]] | None): Optional compression settings
                (default is None).

        Raises:
            ModelNotMatchingError: If the provided model is not an XGBoost sklearn model.

        """
        if not isinstance(model, xgb.XGBModel):
            raise ModelNotMatchingError("Provided model is not an XGBoost sklearn model")
        super().__init__(model, params, num_samples, contributors, additional_info, compression)

    def get_parameters(self) -> dict[str, Any]:
        """
        Extract model parameters as a parsed tree structure.

        Returns:
            Parsed XGBoost JSON structure, or empty dict if model is not fitted.

        """
        try:
            # Get underlying booster and serialize to JSON
            booster = self.model.get_booster()
            model_bytes = booster.save_raw(raw_format="json")
            # Parse JSON to get workable dict structure
            return json.loads(model_bytes.decode("utf-8"))
        except NotFittedError:
            return {}

    def set_parameters(self, params: dict[str, Any] | bytes) -> None:
        """
        Set model parameters from tree structure dict or serialized bytes.

        Args:
            params: Tree structure dict or serialized bytes.

        Raises:
            ModelNotMatchingError: If loading fails.
            TypeError: If params is not a dict after decoding.

        """
        # If bytes, decode compression first
        if isinstance(params, bytes):
            params, additional_info = self.decode_parameters(params)
            self.additional_info.update(additional_info)

        if params is None or len(params) == 0:
            return

        # Type guard: at this point params should be dict
        if not isinstance(params, dict):
            raise TypeError(f"Expected params to be a dict, got {type(params)}")

        # Convert dict to JSON bytes for loading
        model_bytes = json.dumps(params).encode("utf-8")

        # Load into the existing model
        self.model.load_model(bytearray(model_bytes))

    def get_framework(self) -> str:
        """
        Return the framework name identifier.

        Returns:
            str: The framework name ('xgboost').

        """
        return Framework.XGBOOST.value
