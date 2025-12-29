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
"""Manage workflow state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from p2pfl.stages.workflow_type import WorkflowType
from p2pfl.stages.workflows.machine_manager import EventHandlerMachineManager, WorkflowMachineManager

if TYPE_CHECKING:
    from p2pfl.communication.commands.command import Command
    from p2pfl.stages.network_state.network_state import NetworkState
    from p2pfl.stages.workflows.models.event_handler_model import EventHandlerWorkflowModel
    from p2pfl.stages.workflows.models.learning_workflow_model import LearningWorkflowModel

class WorkflowStateManager:
    """Manage the state of the workflows."""

    def __init__(self, workflow_type: WorkflowType) -> None:
        """Initialize the node state."""
        self._workflow_type: WorkflowType = workflow_type

        self._learning_workflow: LearningWorkflowModel | None = None
        self._event_handler_workflow: EventHandlerWorkflowModel | None = None
        self._network_state: NetworkState | None = None
        self._commands: list[Command] = []

    def get_workflow_type(self) -> WorkflowType:
        """Get workflow type."""
        if self._workflow_type is None:
            raise ValueError("Workflow type is not set.")
        return self._workflow_type

    def set_workflow_type(self, workflow_type: WorkflowType) -> None:
        """Set workflow type."""
        self._workflow_type = workflow_type

    def get_learning_workflow(self) -> LearningWorkflowModel:
        """Get learning workflow."""
        if self._learning_workflow is None:
            raise ValueError("Learning workflow is not set.")
        return self._learning_workflow

    def add_learning_workflow(self, learning_workflow: LearningWorkflowModel) -> None:
        """Set learning workflow."""
        if self._workflow_type is None:
            raise ValueError("Workflow type is not set.")
        machine = WorkflowMachineManager().get_machine(self._workflow_type)
        if machine is None:
            raise ValueError("Workflow machine is not set.")
        machine.add_model(learning_workflow, initial="waitingSetup")
        self._learning_workflow = learning_workflow

    def get_event_handler_workflow(self) -> EventHandlerWorkflowModel:
        """Get event handler workflow."""
        if self._event_handler_workflow is None:
            raise ValueError("Event handler workflow is not set.")
        return self._event_handler_workflow

    def add_event_handler_workflow(self, event_handler_workflow: EventHandlerWorkflowModel) -> None:
        """Set event handler workflow."""
        if self._workflow_type is None:
            raise ValueError("Workflow type is not set.")
        machine = EventHandlerMachineManager().get_machine(self._workflow_type)
        if machine is None:
            raise ValueError("Event handler machine is not set.")
        machine.add_model(event_handler_workflow, initial="waitingContextUpdate")
        self._event_handler_workflow = event_handler_workflow

    def get_network_state(self) -> NetworkState:
        """Get network state."""
        if self._network_state is None:
            raise ValueError("Network state is not set.")
        return self._network_state

    def set_network_state(self, network_state: NetworkState) -> None:
        """Set network state."""
        self._network_state = network_state

    def get_commands(self) -> list[Command]:
        """Get commands."""
        return self._commands
    
    def add_commands(self, commands: list[Command]) -> None:
        """Add commands."""
        self._commands.extend(commands)

    def set_commands(self, commands: list[Command]) -> None:
        """Set commands."""
        self._commands = commands
