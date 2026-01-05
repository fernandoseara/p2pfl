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
"""Shared fixtures and mocks for aggregator tests."""

import copy
from typing import Any

import numpy as np

from p2pfl.learning.frameworks.p2pfl_model import TreeBasedModel, WeightBasedModel


class WeightBasedModelMock(WeightBasedModel):
    """Mock WeightBasedModel for testing without ML frameworks."""

    def __init__(
        self,
        params: list[np.ndarray] | None = None,
        num_samples: int = 1,
        contributors: list[str] | None = None,
        additional_info: dict[str, Any] | None = None,
    ) -> None:
        """Initialize mock model."""
        self._params = params
        self._num_samples = num_samples
        self._contributors = contributors or []
        self._additional_info = additional_info or {}

    def get_parameters(self) -> list[np.ndarray]:  # noqa: D102
        return self._params or []

    def set_parameters(self, params: list[np.ndarray]) -> None:  # noqa: D102
        self._params = params

    def get_num_samples(self) -> int:  # noqa: D102
        return self._num_samples

    def add_info(self, key: str, value: Any) -> None:  # noqa: D102
        self._additional_info[key] = value

    def get_info(self, key: str | None = None) -> Any:  # noqa: D102
        if key is None:
            return self._additional_info
        return self._additional_info.get(key, {})

    def build_copy(self, **kwargs) -> "WeightBasedModelMock":  # noqa: D102
        return WeightBasedModelMock(
            params=kwargs.get("params", copy.deepcopy(self._params)),
            num_samples=kwargs.get("num_samples", self._num_samples),
            contributors=kwargs.get("contributors", copy.deepcopy(self._contributors)),
            additional_info=kwargs.get("additional_info", copy.deepcopy(self._additional_info)),
        )

    def get_contributors(self) -> list[str]:  # noqa: D102
        return self._contributors

    def get_framework(self) -> str:  # noqa: D102
        return "mock"


class TreeBasedModelMock(TreeBasedModel):
    """Mock TreeBasedModel for testing without XGBoost."""

    def __init__(
        self,
        params: dict | list | None = None,
        num_samples: int = 1,
        contributors: list[str] | None = None,
        additional_info: dict[str, Any] | None = None,
    ) -> None:
        """Initialize mock model."""
        self._params = params
        self._num_samples = num_samples
        self._contributors = contributors or []
        self._additional_info = additional_info or {}

    def get_parameters(self) -> dict[str, Any]:  # noqa: D102
        if isinstance(self._params, dict):
            return self._params
        return {}

    def set_parameters(self, params: dict | list) -> None:  # noqa: D102
        self._params = params

    def get_num_samples(self) -> int:  # noqa: D102
        return self._num_samples

    def add_info(self, key: str, value: Any) -> None:  # noqa: D102
        self._additional_info[key] = value

    def get_info(self, key: str | None = None) -> Any:  # noqa: D102
        if key is None:
            return self._additional_info
        return self._additional_info.get(key, {})

    def build_copy(self, **kwargs) -> "TreeBasedModelMock":  # noqa: D102
        return TreeBasedModelMock(
            params=kwargs.get("params", copy.deepcopy(self._params)),
            num_samples=kwargs.get("num_samples", self._num_samples),
            contributors=kwargs.get("contributors", copy.deepcopy(self._contributors)),
            additional_info=kwargs.get("additional_info", copy.deepcopy(self._additional_info)),
        )

    def get_contributors(self) -> list[str]:  # noqa: D102
        return self._contributors

    def get_framework(self) -> str:  # noqa: D102
        return "mock_xgboost"
