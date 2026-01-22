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

"""Stage factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from p2pfl.communication.commands.message.asyDFL import (
    IndexInformationUpdatingCommand,
    LossInformationUpdatingCommand,
    ModelInformationUpdatingCommand,
    PushSumWeightInformationUpdatingCommand,
)
from p2pfl.communication.commands.message.node_initialized_command import NodeInitializedCommand
from p2pfl.communication.commands.message.peer_round_updated_command import PeerRoundUpdatedCommand
from p2pfl.learning.frameworks.custom_model_factory import CustomModelFactory
from p2pfl.stages.local_state.async_node_state import AsyncLocalNodeState
from p2pfl.stages.network_state.async_network_state import AsyncNetworkState
from p2pfl.stages.workflow_type import WorkflowType
from p2pfl.stages.workflows.builder.workflow_builder import WorkflowBuilder
from p2pfl.stages.workflows.models.asyncFL.async_learning_workflow import AsyncLearningWorkflowModel
from p2pfl.stages.workflows.models.asyncFL.async_learning_workflow import get_states as get_learning_states
from p2pfl.stages.workflows.models.asyncFL.async_learning_workflow import get_transitions as get_learning_transitions
from p2pfl.stages.workflows.workflow_state_manager import WorkflowStateManager
from p2pfl.utils.pytransitions import TimeoutMachine

if TYPE_CHECKING:
    from p2pfl.communication.commands.command import Command
    from p2pfl.node import Node


class AsyDFLWorkflowBuilder(WorkflowBuilder):
    """Builder for AsyDFL workflows."""

    def __init__(self):
        """Initialize the workflow builder."""
        self.reset()

    @property
    def workflow_state_manager(self) -> WorkflowStateManager:
        """Get the resulting workflow."""
        return self._workflow_state_manager

    def reset(self) -> None:
        """Reset the builder to initial state."""
        self._states: list = []
        self._transitions: list = []
        self._workflow_type: WorkflowType = WorkflowType.ASYNC

        self._network_state: AsyncNetworkState | None = None
        self._local_state: AsyncLocalNodeState | None = None

        self._workflow_state_manager: WorkflowStateManager = WorkflowStateManager(self._workflow_type)

    def create_states(self) -> None:
        """Create states."""
        self._states = get_learning_states()

    def create_transitions(self) -> None:
        """Create transitions."""
        self._transitions = get_learning_transitions()

    def create_local_state(self, node: Node) -> None:
        """Create local state."""
        self._local_state = AsyncLocalNodeState(node.address)
        self._workflow_state_manager.set_local_state(self._local_state)

    def create_network_state(self) -> None:
        """Create network state."""
        self._network_state = AsyncNetworkState()
        self._workflow_state_manager.set_network_state(self._network_state)

    def create_workflow_machine(self) -> None:
        """Create the workflow machine."""
        self._workflow_state_manager.set_workflow_machine(
            TimeoutMachine(
                model=None,
                states=self._states,
                transitions=self._transitions,
                initial=None,
                queued="model",
                ignore_invalid_triggers=True,
                model_override=True,
                finalize_event="finalize_logging",
            )
        )

    def create_training_workflow_model(self, node: Node) -> None:
        """Create a workflow model."""
        if self._network_state is None:
            raise ValueError("Network state is not set.")
        if self._local_state is None:
            raise ValueError("Local state is not set.")

        self._workflow_state_manager.add_learning_workflow(
            AsyncLearningWorkflowModel(node, self._local_state, self._network_state), initialState="waitingSetup"
        )

    def create_commands(self, node: Node) -> None:
        """Create commands list."""
        commands: list[Command] = [
            NodeInitializedCommand(node),
            PeerRoundUpdatedCommand(node),
            LossInformationUpdatingCommand(node),
            IndexInformationUpdatingCommand(node),
            ModelInformationUpdatingCommand(node),
            PushSumWeightInformationUpdatingCommand(node),
        ]
        self._workflow_state_manager.add_commands(commands)
        node.get_communication_protocol().add_command(commands)

    def create_model(self, node: Node) -> None:
        """Create model."""
        model = node.get_learner().get_model()
        node.get_learner().set_model(CustomModelFactory.create_model("AsyDFL", model))
