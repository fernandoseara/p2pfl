#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2022 Pedro Guijas Bravo.
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
"""Workflows."""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from typing import Any

from p2pfl.stages.workflows.models.workflow_model import WorkflowModel


class LearningWorkflowModel(WorkflowModel):
    """Base for the training workflow."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the learning workflow model."""
        self._background_tasks: set[asyncio.Task[Any]] = set()
        super().__init__(*args, **kwargs)

    @abstractmethod
    async def is_finished(self) -> bool:
        """Check if the workflow has finished."""
        ...

    @abstractmethod
    async def node_started(self, source: str) -> bool:
        """Handle peer started notification."""
        ...

    ########################################
    # EVENTS (Overridden by pytransitions) #
    ########################################
    # These methods are dynamically created by pytransitions based on
    # the transitions defined in the state machine.

    async def next_stage(self) -> bool:
        """Handle the next stage event."""
        raise NotImplementedError

    async def setup(
        self,
        is_initiator: bool,
        experiment_name: str,
        rounds: int,
        epochs: int,
        trainset_size: int,
    ) -> bool:
        """Handle the setup event."""
        raise NotImplementedError

    async def stop_learning(self) -> bool:
        """Handle the stop learning event."""
        raise NotImplementedError
