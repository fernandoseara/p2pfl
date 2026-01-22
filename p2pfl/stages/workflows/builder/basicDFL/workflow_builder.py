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

from p2pfl.communication.commands.message.models_agregated_command import ModelsAggregatedCommand
from p2pfl.communication.commands.message.node_initialized_command import NodeInitializedCommand
from p2pfl.communication.commands.message.peer_round_updated_command import PeerRoundUpdatedCommand
from p2pfl.communication.commands.message.vote_train_set_command import VoteTrainSetCommand
from p2pfl.communication.commands.weights.full_model_command import FullModelCommand
from p2pfl.communication.commands.weights.partial_model_command import PartialModelCommand
from p2pfl.stages.local_state.dfl_node_state import DFLLocalNodeState
from p2pfl.stages.network_state.basic_network_state import BasicNetworkState
from p2pfl.stages.workflow_type import WorkflowType
from p2pfl.stages.workflows.builder.workflow_builder import WorkflowBuilder
from p2pfl.stages.workflows.models.basicDFL.basic_learning_workflow import BasicLearningWorkflowModel
from p2pfl.stages.workflows.models.basicDFL.basic_learning_workflow import get_states as get_learning_states
from p2pfl.stages.workflows.models.basicDFL.basic_learning_workflow import get_transitions as get_learning_transitions
from p2pfl.stages.workflows.workflow_state_manager import WorkflowStateManager
from p2pfl.utils.pytransitions import TimeoutMachine

if TYPE_CHECKING:
    from p2pfl.communication.commands.command import Command
    from p2pfl.node import Node


class BasicDFLWorkflowBuilder(WorkflowBuilder):
    """Builder for BasicDFL workflows."""

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
        self._workflow_type: WorkflowType = WorkflowType.BASIC

        self._network_state: BasicNetworkState | None = None
        self._local_state: DFLLocalNodeState | None = None

        self._workflow_state_manager: WorkflowStateManager = WorkflowStateManager(self._workflow_type)

    def create_states(self) -> None:
        """Create states."""
        self._states = get_learning_states()

    def create_transitions(self) -> None:
        """Create transitions."""
        self._transitions = get_learning_transitions()

    def create_local_state(self, node: Node) -> None:
        """Create local state."""
        self._local_state = DFLLocalNodeState(node.address)
        self._workflow_state_manager.set_local_state(self._local_state)

    def create_network_state(self) -> None:
        """Create network state."""
        self._network_state = BasicNetworkState()
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
            BasicLearningWorkflowModel(node, self._local_state, self._network_state), initialState="waitingSetup"
        )

    def create_commands(self, node: Node) -> None:
        """Create commands list."""
        commands: list[Command] = [
            NodeInitializedCommand(node),
            PeerRoundUpdatedCommand(node),
            VoteTrainSetCommand(node),
            ModelsAggregatedCommand(node),
            PartialModelCommand(node),
            FullModelCommand(node),
        ]
        self._workflow_state_manager.add_commands(commands)
        node.get_communication_protocol().add_command(commands)
