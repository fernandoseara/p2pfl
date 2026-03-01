#
# This file is part of the p2pfl (see https://github.com/pguijas/p2pfl).
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
"""Experiment class."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from p2pfl.exceptions import ZeroRoundsException
from p2pfl.management.logger import logger


@dataclass
class Experiment:
    """
    Tracks experiment metadata and round progression.

    TODO: Integrate with a dynamic log system so that all experiment state
    changes (round progression, metric updates, data dict mutations) are
    automatically emitted as log events. The log system should be pluggable
    to support different backends (file, MLflow, wandb, etc.).

    Args:
        exp_name: The name of the experiment.
        total_rounds: The total rounds of the experiment.
        is_initiator: Whether this node initiated the experiment.
        epochs_per_round: The number of epochs per round.
        round: The current round number.
        data: Flexible tracking data for MLOps persistence.
        trainset_size: The number of nodes in the training set.
        dataset_name: The name of the dataset.
        model_name: The name of the model.
        aggregator_name: The name of the aggregator.
        framework_name: The name of the framework.
        learning_rate: The learning rate.
        batch_size: The batch size.

    """

    exp_name: str
    total_rounds: int
    workflow: str = "basic"
    is_initiator: bool = False
    epochs_per_round: int = 1
    round: int = 0

    data: dict[str, Any] = field(default_factory=dict)
    trainset_size: int | None = None
    dataset_name: str | None = None
    model_name: str | None = None
    aggregator_name: str | None = None
    framework_name: str | None = None
    learning_rate: float | None = None
    batch_size: int | None = None

    def __post_init__(self) -> None:
        """Validate experiment configuration."""
        if self.total_rounds < 1:
            raise ZeroRoundsException("Rounds must be greater than 0.")

    @classmethod
    def create(cls, **kwargs: Any) -> Experiment:
        """Create an Experiment, routing unknown kwargs to ``data``."""
        known = {f.name for f in dataclasses.fields(cls)}
        init_kwargs = {k: v for k, v in kwargs.items() if k in known}
        extra = {k: v for k, v in kwargs.items() if k not in known}
        if extra:
            init_kwargs.setdefault("data", {}).update(extra)
        return cls(**init_kwargs)

    def is_complete(self) -> bool:
        """Check whether the experiment has reached its total round count."""
        return self.round >= self.total_rounds

    def increase_round(self, address: str) -> None:
        """Increment the round counter and notify the logger."""
        self.round += 1
        logger.round_updated(address, self.round)

    def to_dict(self, exclude_none: bool = True) -> dict:
        """
        Convert the experiment to a dictionary.

        Args:
            exclude_none: If True, exclude fields with None values.

        Returns:
            Dictionary representation of the experiment.

        """
        config = {
            "exp_name": self.exp_name,
            "total_rounds": self.total_rounds,
            "workflow": self.workflow,
            "epochs_per_round": self.epochs_per_round,
            "trainset_size": self.trainset_size,
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "aggregator_name": self.aggregator_name,
            "framework_name": self.framework_name,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
        }

        if exclude_none:
            return {k: v for k, v in config.items() if v is not None}
        return config

    def __str__(self) -> str:
        """Return the string representation of the experiment."""
        metadata_str = f", epochs_per_round={self.epochs_per_round}"
        if self.dataset_name:
            metadata_str += f", dataset_name={self.dataset_name}"
        if self.model_name:
            metadata_str += f", model_name={self.model_name}"
        if self.aggregator_name:
            metadata_str += f", aggregator_name={self.aggregator_name}"
        if self.framework_name:
            metadata_str += f", framework_name={self.framework_name}"
        if self.learning_rate is not None:
            metadata_str += f", learning_rate={self.learning_rate}"
        if self.batch_size is not None:
            metadata_str += f", batch_size={self.batch_size}"

        return f"Experiment(exp_name={self.exp_name}, total_rounds={self.total_rounds}{metadata_str})"
