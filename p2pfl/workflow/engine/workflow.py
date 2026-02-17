#
# This file is part of the federated_learning_p2p (p2pfl) distribution
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
"""Workflow base class."""

from __future__ import annotations

import asyncio
import contextlib
import enum
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from p2pfl.communication.commands.workflow import MessageCommand, WeightsCommand
from p2pfl.management.logger import logger
from p2pfl.workflow import CommandEntry
from p2pfl.workflow.engine.experiment import Experiment
from p2pfl.workflow.factory import WorkflowType

if TYPE_CHECKING:
    from p2pfl.node import Node


class WorkflowStatus(enum.Enum):
    """Status of a workflow run."""

    IDLE = "idle"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    ERROR = "error"

    @property
    def is_finished(self) -> bool:
        """Check if the workflow finished successfully."""
        return self == WorkflowStatus.FINISHED

    @property
    def is_terminal(self) -> bool:
        """Check if the workflow reached any conclusive state."""
        return self in (WorkflowStatus.FINISHED, WorkflowStatus.CANCELLED, WorkflowStatus.ERROR)


class Workflow:
    """Base class for learning workflows."""

    ###########################
    #  Initialization & State #
    ###########################

    # Used by pytransitions
    state: str
    # Populated by @on_message decorator via __set_name__
    _message_registry: dict[str, CommandEntry]

    def __init__(self, node: Node) -> None:
        """Initialize the workflow."""
        self.node: Node = node
        self.experiment: Experiment | None = None
        self.round: int = 0
        self.status: WorkflowStatus = WorkflowStatus.IDLE
        self.error: Exception | None = None
        self._task: asyncio.Task[None] | None = None

    def increase_round(self) -> None:
        """Increment the round counter and notify the logger."""
        self.round += 1
        logger.round_updated(self.node.address, self.round)

    ###########################
    #  Status                 #
    ###########################

    @property
    @abstractmethod
    def workflow_type(self) -> WorkflowType:
        """Return the workflow type."""
        ...

    def mark_finished(self) -> None:
        """Mark the workflow as finished successfully."""
        self.status = WorkflowStatus.FINISHED

    ###########################
    #  Lifecycle (start/stop) #
    ###########################

    async def start(
        self,
        experiment_name: str,
        rounds: int,
        epochs: int,
        is_initiator: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Set up and run the learning loop in a background task.

        Args:
            experiment_name: The name of the experiment.
            rounds: Number of rounds.
            epochs: Number of epochs per round.
            is_initiator: Whether this node initiated the learning.
            **kwargs: Workflow-specific parameters (e.g. trainset_size for BasicDFL).

        """
        self.status = WorkflowStatus.RUNNING

        # Configure run parameters before the state machine starts
        self.experiment = Experiment(
            experiment_name,
            rounds,
            is_initiator=is_initiator,
            epochs_per_round=epochs,
            **kwargs,
        )
        self.round = 0
        logger.experiment_started(self.node.address, self.experiment)
        self.node.learner.set_epochs(epochs)
        try:
            self._register_commands()
            self._task = asyncio.create_task(self._execute())
        except Exception:
            self._cleanup()
            raise

    async def stop(self) -> None:
        """Stop the workflow and cancel the background task."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None

    async def _execute(self) -> None:
        """Wrap _run with error handling and _cleanup."""
        try:
            await self._run()
            logger.info(self.node.address, "✅ Learning finished.")
        except asyncio.CancelledError:
            if not self.status.is_finished:
                self.status = WorkflowStatus.CANCELLED
            logger.info(self.node.address, "Learning cancelled.")
            raise
        except Exception as e:
            self.status = WorkflowStatus.ERROR
            self.error = e
            logger.error(self.node.address, f"Learning failed: {e}")
            raise
        finally:
            self._cleanup()

    @abstractmethod
    async def _run(self) -> None:
        """Run the workflow. Subclasses implement their own loop and stop logic."""
        ...

    def _cleanup(self) -> None:
        """Unregister commands and clean up. Subclasses should call super()."""
        self._unregister_commands()
        self.experiment = None
        self.round = 0

    ##############################
    #  Command Registration      #
    ##############################

    def get_message_registry(self) -> dict[str, CommandEntry]:
        """Get the merged message registry across the MRO chain."""
        merged: dict[str, CommandEntry] = {}
        for cls in reversed(type(self).__mro__):
            if "_message_registry" in cls.__dict__:
                merged.update(cls.__dict__["_message_registry"])
        return merged

    def _register_commands(self) -> None:
        """Register workflow commands from get_message_registry() with the protocol."""
        protocol = self.node.communication_protocol
        for cmd_name, entry in self.get_message_registry().items():
            if entry.is_weights:
                protocol.add_command(WeightsCommand(self.node, cmd_name))
            else:
                protocol.add_command(MessageCommand(self.node, cmd_name))

    def _unregister_commands(self) -> None:
        """Unregister workflow commands from the protocol."""
        protocol = self.node.communication_protocol
        for cmd_name in self.get_message_registry():
            protocol.remove_command(cmd_name)
