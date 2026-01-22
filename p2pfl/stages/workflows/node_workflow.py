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
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from p2pfl.communication.commands.message.stop_learning_command import StopLearningCommand
from p2pfl.management.logger import logger
from p2pfl.stages.workflow_type import WorkflowType
from p2pfl.stages.workflows.builder.workflow_builder import WorkflowBuilderFactory
from p2pfl.stages.workflows.builder.workflow_director import WorkflowDirector
from p2pfl.stages.workflows.workflow_state_manager import WorkflowStateManager
from p2pfl.utils.asyncio import sync_or_async
from p2pfl.utils.pytransitions import StateAdapter, TimeoutMachine, TransitionAdapter
from p2pfl.utils.singleton import SingletonMeta

# Set transitions' log level
logging.getLogger("transitions").setLevel(logging.ERROR)

if TYPE_CHECKING:
    from transitions import Machine

    from p2pfl.node import Node
    from p2pfl.stages.local_state.node_state import LocalNodeState
    from p2pfl.stages.network_state.network_state import NetworkState
    from p2pfl.stages.workflows.models.learning_workflow_model import LearningWorkflowModel


def get_states() -> list[dict]:
    """Get the states for the NodeWorkflow."""
    states: list[StateAdapter] = [
        # Initial state
        StateAdapter(name="stopped"),
        StateAdapter(name="waitingForLearningStart"),
        # Learning state
        StateAdapter(name="learning"),
        # Final state
        StateAdapter(name="learningFinished", final=True),
    ]

    return [state.to_dict() for state in states]


def get_transitions() -> list[dict]:
    """Get the transitions for the NodeWorkflow."""
    transitions: list[TransitionAdapter] = [
        TransitionAdapter(trigger="start_node", source="stopped", dest="waitingForLearningStart", before="start"),
        # Starting the learning process
        TransitionAdapter(trigger="start_learning", source="waitingForLearningStart", dest="learning", before="set_initiator"),
        TransitionAdapter(trigger="peer_learning_initiated", source="waitingForLearningStart", dest="learning"),
        # Learning process finished
        TransitionAdapter(trigger="learning_finished", source="learning", dest="learningFinished"),
        # Stopping the workflow
        TransitionAdapter(trigger="stop_node", source="*", dest="stopped", before="stop"),
        # Communications
        TransitionAdapter(
            trigger="connect_node", source=["waitingForLearningStart", "learning", "learningFinished"], dest=None, before="connect"
        ),
        TransitionAdapter(
            trigger="disconnect_node", source=["waitingForLearningStart", "learning", "learningFinished"], dest=None, before="disconnect"
        ),
    ]

    return [transition.to_dict() for transition in transitions]


class NodeWorkflowModel:
    """Base for the node workflow."""

    state: str

    def __init__(self, node: Node, state_history_length: int = 10):
        """Initialize the workflow model."""
        self._candidates: list = []
        self._background_tasks: set[asyncio.Task] = set()

        self.node = node
        self.is_initiator: bool = False

        # Error tracking
        self.failed: bool = False
        self.error: Exception | None = None

        self.workflow_state_manager: WorkflowStateManager | None = None
        WorkflowMachineManager().add_model(self)

    def __remove__(self):
        """Cleanup the workflow model."""
        WorkflowMachineManager().remove_model(self)

    #################
    #    Getters    #
    #################
    def get_learning_workflow(self) -> LearningWorkflowModel:
        """
        Get the learning workflow.

        Returns:
            The current learning workflow of the node.

        """
        if self.workflow_state_manager is None:
            raise RuntimeError("Workflow is not initialized")
        return self.workflow_state_manager.get_learning_workflow()

    def get_workflow_type(self) -> WorkflowType:
        """
        Get the workflow type.

        Returns:
            The current workflow type of the node.

        """
        if self.workflow_state_manager is None:
            raise RuntimeError("Workflow is not initialized")
        return self.workflow_state_manager.get_workflow_type()

    def get_commands(self) -> list:
        """
        Get the commands list.

        Returns:
            The current commands list of the node.

        """
        if self.workflow_state_manager is None:
            raise RuntimeError("Workflow is not initialized")
        return self.workflow_state_manager.get_commands()

    def get_local_state(self) -> LocalNodeState:
        """
        Get the local state.

        Returns:
            The current local state.

        """
        if self.workflow_state_manager is None:
            raise RuntimeError("Workflow is not initialized")
        return self.workflow_state_manager.get_local_state()

    def get_network_state(self) -> NetworkState:
        """
        Get the network state.

        Returns:
            The current network state.

        """
        if self.workflow_state_manager is None:
            raise RuntimeError("Workflow is not initialized")
        return self.workflow_state_manager.get_network_state()

    ########################################
    # EVENTS (Overridden by pytransitions) #
    ########################################
    async def start_node(self, wait: bool = False) -> None:
        """
        Start the node: server and neighbors (gossip and heartbeat).

        Args:
            wait: If True, the function will wait until the server is terminated.

        """
        raise RuntimeError("Should be overridden!")

    async def stop_node(self) -> None:
        """Stop the node."""
        raise RuntimeError("Should be overridden!")

    def connect_node(self, addr: str) -> bool:
        """
        Connect a node to another.

        Warning:
            Adding nodes while learning is running is not fully supported.

        Args:
            addr: The address of the node to connect to.

        Returns:
            True if the node was connected, False otherwise.

        """
        raise RuntimeError("Should be overridden!")

    def disconnect_node(self, addr: str) -> None:
        """
        Disconnects a node from another.

        Args:
            addr: The address of the node to disconnect from.

        """
        raise RuntimeError("Should be overridden!")

    async def start_learning(
        self,
        workflow_type: WorkflowType,
        experiment_name: str,
        rounds: int,
        epochs: int,
        trainset_size: int,
    ) -> bool:
        """Handle the start learning event."""
        raise RuntimeError("Should be overridden!")

    async def peer_learning_initiated(
        self, workflow_type: WorkflowType, experiment_name: str, rounds: int, epochs: int, trainset_size: int, source: str
    ) -> bool:
        """Handle the peer learning initiated event."""
        raise RuntimeError("Should be overridden!")

    async def learning_finished(self) -> bool:
        """Handle the learning finished event."""
        raise RuntimeError("Should be overridden!")

    ##############
    # PROPERTIES #
    ##############

    # @property
    # def state(self):
    #     """Get the current state of the workflow."""
    #     return self.state_log[-1]

    # @state.setter
    # def state(self, value):
    #     """Set the current state of the workflow."""
    #     self.state_log.append(value)

    @property
    def waiting_for_learning_start(self) -> bool:
        """
        Check if the workflow is waiting for the learning start.

        Returns:
            bool: True if the workflow is waiting for the learning start, False otherwise.

        """
        return self.state == "waitingForLearningStart"

    @property
    def running(self) -> bool:
        """
        Check if the workflow is running.

        Returns:
            bool: True if the workflow is running, False otherwise.

        """
        return self.state != "stopped"

    @property
    def finished(self) -> bool:
        """
        Check if the workflow is finished.

        Returns:
            bool: True if the workflow is finished, False otherwise.

        """
        return self.state == "learningFinished"

    async def set_initiator(self, *args, **kwargs) -> None:
        """Set the model initialized."""
        # Set the initiator flag in the network state
        self.is_initiator = True

    ###################
    # STATE CALLBACKS #
    ###################
    async def on_enter_learning(self, workflow_type: WorkflowType, *args, **kwargs):
        """Start the training."""
        logger.info(self.node.address, f"⏳ Setting environment for learning. Learning type: {workflow_type.value}")

        # Workflow factory
        self.workflow_type = workflow_type
        director = WorkflowDirector()
        workflow_builder = WorkflowBuilderFactory.get_builder(workflow_type.value)()
        director.builder = workflow_builder

        # Create workflow
        director.build_workflow_state_manager(self.node)
        self.workflow_state_manager = workflow_builder.workflow_state_manager

        # Start learning
        if self.workflow_state_manager is None:
            raise RuntimeError("Workflow is not initialized")
        await self.get_learning_workflow().setup(self.is_initiator, *args, **kwargs)

        # Run learning in background
        await self._fire_task(self.run)

    async def on_enter_learningFinished(self, *args, **kwargs) -> None:
        """Finish the learning."""
        logger.info(self.node.address, "✅ Learning finished.")
        if self.workflow_state_manager is None:
            raise RuntimeError("Workflow is not initialized")
        self.workflow_state_manager.remove_learning_workflow()
        self.workflow_state_manager = None

    async def run(self) -> None:
        """Run the learning workflow."""
        learning_workflow = self.get_learning_workflow()

        # Loop until the learning workflow is finished
        while await learning_workflow.is_finished() is False:
            await learning_workflow.next_stage()
            await asyncio.sleep(0.1)

        await self.learning_finished()

    ##############
    # START NODE #
    ##############
    async def start(self, wait: bool = False) -> None:
        """
        Start the node: server and neighbors (gossip and heartbeat).

        Args:
            wait: If True, the function will wait until the server is terminated.

        """
        logger.info(self.node.address, "🚀 Starting node.")

        # P2PFL Web Services
        logger.register_node(self.node.address)
        # Communication Protocol
        await self.node.get_communication_protocol().start()
        if wait:
            self.node.get_communication_protocol().wait_for_termination()
            logger.info(self.node.address, "gRPC terminated.")

    #################
    # COMMUNICATION #
    #################
    @sync_or_async
    async def connect(self, addr: str) -> bool:
        """
        Connect a node to another.

        Warning:
            Adding nodes while learning is running is not fully supported.

        Args:
            addr: The address of the node to connect to.

        Returns:
            True if the node was connected, False otherwise.

        """
        return await self.node.communication_protocol.connect(addr)

    @sync_or_async
    async def disconnect(self, addr: str) -> None:
        """
        Disconnects a node from another.

        Args:
            addr: The address of the node to disconnect from.

        """
        logger.info(self.node.address, f"Removing {addr}...")
        await self.node.communication_protocol.disconnect(addr)

    #############
    # INTERRUPT #
    #############
    async def stop(self) -> None:
        """Stop the node: server and neighbors (gossip and heartbeat)."""
        logger.info(self.node.address, "🛑 Stopping node.")
        communication_protocol = self.node.get_communication_protocol()
        try:
            # Notify neighbors
            await communication_protocol.broadcast_gossip(communication_protocol.build_msg(StopLearningCommand.get_name()))

            # Stop server
            await communication_protocol.stop()
            # self.node.get_communication_protocol().remove_command(self.node.workflow_factory.create_commands(self.node))

            await self.get_learning_workflow().stop_learning()

            # Unregister node
            logger.unregister_node(self.node.address)
        except Exception:
            pass

    async def interrupt(self) -> None:
        """Interrupt the workflow."""
        await asyncio.sleep(1)
        machine = WorkflowMachineManager().get_machine()
        if machine is not None:
            for task in machine.async_tasks[id(self)]:
                task.cancel()
            machine._transition_queue_dict[id(self)].clear()

        await self.stop()

    ######################
    # LOGGING CALLBACKS #
    ######################
    def finalize_logging(self, *args, **kwargs) -> None:
        """Log callback."""
        logger.debug(self.node.address, f"🏃 Running stage: {(self.state)}")

    def test(self, *args, **kwargs) -> None:
        """Test function for debugging."""
        logger.info(self.node.address, "Test function called.")

    async def _fire_task(self, func: Callable, *args, **kwargs) -> None:
        """Schedule a state machine trigger as a background task."""
        task = asyncio.create_task(func(*args, **kwargs))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)


class WorkflowMachineManager(metaclass=SingletonMeta):
    """Manager for workflow state machines."""

    def __init__(self) -> None:
        """Initialize the node state."""
        self._machine: Machine = TimeoutMachine(
            states=get_states(),
            transitions=get_transitions(),
            initial="stopped",
            queued="model",
            ignore_invalid_triggers=True,
            finalize_event="finalize_logging",
            model_override=True,
        )

    def get_machine(self) -> Machine | None:
        """Get the state machine for a given workflow type."""
        return self._machine

    def add_model(self, model: NodeWorkflowModel) -> None:
        """Add a model to the state machine."""
        self._machine.add_model(model)

    def remove_model(self, model: NodeWorkflowModel) -> None:
        """Remove the model from the state machine."""
        self._machine.remove_model(model)
