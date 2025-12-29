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
"""Workflows machine manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

from transitions.extensions.nesting import NestedState

from p2pfl.management.logger import logger
from p2pfl.stages.workflow_type import WorkflowType
from p2pfl.utils.singleton import SingletonMeta

if TYPE_CHECKING:
    from transitions import Machine

class WorkflowMachineManager(metaclass=SingletonMeta):
    """Manager for workflow state machines."""

    machines: dict[WorkflowType, Machine]

    def __init__(self, machines: dict[WorkflowType, Machine] | None = None) -> None:
        """Initialize the node state."""
        self.machines: dict[WorkflowType, Machine] = machines or {}

    def get_machine(self, workflow_type: WorkflowType) -> Machine | None:
        """Get the state machine for a given workflow type."""
        key = workflow_type
        return self.machines.get(key, None)

    def add_machine(self, workflow_type: WorkflowType, machine: Machine) -> None:
        """Add a state machine for a given workflow type."""
        key = workflow_type
        self.machines[key] = machine

    def remove_machine(self, workflow_type: WorkflowType) -> None:
        """Remove the state machine for a given workflow type."""
        key = workflow_type
        if key in self.machines:
            del self.machines[key]

    def clear_machines(self) -> None:
        """Clear all state machines."""
        self.machines.clear()

class EventHandlerMachineManager(metaclass=SingletonMeta):
    """Manager for event handler state machines."""

    def __init__(self) -> None:
        """Initialize the node state."""
        self._machines: dict[WorkflowType, Machine] = {}

    def get_machine(self, workflow_type: WorkflowType) -> Machine | None:
        """Get the state machine for a given workflow type."""
        key = workflow_type
        return self._machines.get(key, None)

    def add_machine(self, workflow_type: WorkflowType, machine: Machine) -> None:
        """Add a state machine for a given workflow type."""
        key = workflow_type
        self._machines[key] = machine

    def remove_machine(self, workflow_type: WorkflowType) -> None:
        """Remove the state machine for a given workflow type."""
        key = workflow_type
        if key in self._machines:
            del self._machines[key]

    def clear_machines(self) -> None:
        """Clear all state machines."""
        self._machines.clear()
