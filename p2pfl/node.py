#
# This file is part of the federated_learning_p2p (p2pfl) distribution (see https://github.com/pguijas/p2pfl).
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

"""P2PFL Node."""

import asyncio
import os
import random
import time
import traceback
from typing import Any, Dict, Optional

from p2pfl.communication.commands.message.start_learning_command import StartLearningCommand
from p2pfl.communication.commands.message.stop_learning_command import StopLearningCommand
from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
from p2pfl.communication.protocols.protobuff.grpc import GrpcCommunicationProtocol
from p2pfl.exceptions import LearnerRunningException, NodeRunningException, ZeroRoundsException
from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.aggregators.fedavg import FedAvg
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.learning.frameworks.learner_factory import LearnerFactory
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.learning.frameworks.simulation import try_init_learner_with_ray
from p2pfl.management.logger import logger
from p2pfl.node_state import LocalNodeState
from p2pfl.settings import Settings
from p2pfl.stages.workflow_factory import WorkflowFactory, WorkflowFactoryProducer
from p2pfl.stages.workflow_type import WorkflowType
from p2pfl.stages.workflows.models.basic_learning_workflow_model import LearningWorkflowModel
from p2pfl.stages.workflows.workflows import LearningWorkflow
from p2pfl.utils.asyncio import dualmethod

# Disbalbe grpc log (pytorch causes warnings)
if logger.get_level_name(logger.get_level()) != "DEBUG":
    os.environ["GRPC_VERBOSITY"] = "NONE"


class Node:
    """
    Represents a learning node in the federated learning network.

    The following example shows how to create a node with a MLP model and a MnistFederatedDM dataset. Then, the node is
    started, connected to another node, and the learning process is started.

    >>> node = Node(
    ...     MLP(),
    ...     MnistFederatedDM(),
    ... )
    >>> node.start()
    >>> node.connect("127.0.0.1:666")
    >>> node.set_start_learning(rounds=2, epochs=1)

    Args:
        model: Model to be used in the learning process.
        data: Dataset to be used in the learning process.
        addr: The address of the node.
        learner: The learner class to be used.
        aggregator: The aggregator class to be used.
        protocol: The communication protocol to be used.
        **kwargs: Additional arguments.

    .. todo::
        Instanciate the aggregator dynamically.

    .. todo::
        Connect nodes dynamically (while learning).

    """

    def __init__(
        self,
        model: P2PFLModel,
        data: P2PFLDataset,
        address: str = "",
        learner: Optional[Learner] = None,
        aggregator: Optional[Aggregator] = None,
        protocol: Optional[CommunicationProtocol] = None,
        simulation: bool = False,
        **kwargs,
    ) -> None:
        """Initialize a node."""
        # Communication protocol
        self.communication_protocol = GrpcCommunicationProtocol() if protocol is None else protocol
        address = self.communication_protocol.set_addr(address)

        #self.communication_protocol.add_command(commands)

        # Aggregator
        self.aggregator = FedAvg() if aggregator is None else aggregator
        self.aggregator.set_addr(address)

        # Learner
        if learner is None:  # if no learner, use factory default
            learner = LearnerFactory.create_learner(model)()
        self.learner = try_init_learner_with_ray(learner)
        self.learner.set_addr(address)
        self.learner.set_P2PFLModel(model)
        self.learner.set_data(data)
        self.learner.indicate_aggregator(self.aggregator)

        # Simulation
        self.generator: random.Random = random.Random(Settings.general.SEED)

        # State
        self.local_state = LocalNodeState(address)

        # Workflow
        self.workflow_factory: type[WorkflowFactory] | None = None
        self.learning_workflow: LearningWorkflowModel = LearningWorkflowModel(self)
        self.learning_machine: LearningWorkflow = LearningWorkflow(self.learning_workflow)

        # Commands
        self.communication_protocol.add_command([
            StartLearningCommand(self),
            StopLearningCommand(self),
        ])

        self.running = False  # Node state


    #############################
    #  Neighborhood management  #
    #############################

    def connect(self, addr: str) -> bool:
        """
        Connect a node to another.

        Warning:
            Adding nodes while learning is running is not fully supported.

        Args:
            addr: The address of the node to connect to.

        Returns:
            True if the node was connected, False otherwise.

        """
        # Check running
        if not self.is_running():
            raise NodeRunningException("Node is not running.")

        # Connect
        return self.communication_protocol.connect(addr)

    def get_neighbors(self, only_direct: bool = False) -> Dict[str, Any]:
        """
        Return the neighbors of the node.

        Args:
            only_direct: If True, only the direct neighbors will be returned.

        Returns:
            The list of neighbors.

        """
        return self.communication_protocol.get_neighbors(only_direct)

    def disconnect(self, addr: str) -> None:
        """
        Disconnects a node from another.

        Args:
            addr: The address of the node to disconnect from.

        """
        # Check running
        if not self.is_running():
            raise NodeRunningException("Node is not running.")

        # Disconnect
        logger.info(self.address, f"Removing {addr}...")
        self.communication_protocol.disconnect(addr, disconnect_msg=True)

    #######################################
    #   Node Management (servicer loop)   #
    #######################################

    """
    -> reemplazarlo por un decorador (y esto creo que se puede reemplazar por el estado del comm proto -> incluso importarlo de ahí)
    """
    def is_running(self) -> bool:
        """
        Check if the node is running.

        Returns:
            True if the node is running, False otherwise.

        """
        return self.running

    async def start(self, wait: bool = False) -> None:
        """
        Start the node: server and neighbors(gossip and heartbeat).

        Args:
            wait: If True, the function will wait until the server is terminated.

        Raises:
            NodeRunningException: If the node is already running.

        """
        # Check not running
        if self.is_running():
            raise NodeRunningException("Node already running.")

        # P2PFL Web Services
        logger.register_node(self.address)
        # Communication Protocol
        await self.communication_protocol.start()
        if wait:
            self.communication_protocol.wait_for_termination()
            logger.info(self.address, "gRPC terminated.")

        self.running = True

    async def stop(self) -> None:
        """
        Stop the node: server and neighbors(gossip and heartbeat).

        Raises:
            NodeRunningException: If the node is not running.

        """
        logger.info(self.address, "🛑 Stopping node...")
        try:
            # Stop server
            await self.communication_protocol.stop()
            # State
            self.local_state.clear()
            # Unregister node
            logger.unregister_node(self.address)
        except Exception:
            pass

        self.running = False

    ##########################
    #    Learning Setters    #
    ##########################

    def set_learner(self, learner: Learner) -> None:
        """
        Set the learner to be used in the learning process.

        Args:
            learner: The learner to be used in the learning process.

        Raises:
            LearnerRunningException: If the learner is already set.

        """
        if self.local_state.round is not None:
            raise LearnerRunningException("Learner cannot be set after learning is started.")
        self.learner = learner

    def set_model(self, model: P2PFLModel) -> None:
        """
        Set the model to be used in the learning process (by the learner).

        Args:
            model: Model to be used in the learning process.

        Raises:
            LearnerRunningException: If the learner is already set.

        """
        if self.local_state.round is not None:
            raise LearnerRunningException("Data cannot be set after learner is set.")
        self.learner.set_P2PFLModel(model)

    def set_data(self, data: P2PFLDataset) -> None:
        """
        Set the data to be used in the learning process (by the learner).

        Args:
            data: Dataset to be used in the learning process.

        Raises:
            LearnerRunningException: If the learner is already set.

        """
        # Cannot change during training (raise)
        if self.local_state.round is not None:
            raise LearnerRunningException("Data cannot be set after learner is set.")
        self.learner.set_data(data)

    ##########################
    #    Learning Getters    #
    ##########################

    def get_model(self) -> P2PFLModel:
        """
        Get the model.

        Returns:
            The current model of the node.

        """
        return self.learner.get_P2PFLModel()

    def get_data(self) -> P2PFLDataset:
        """
        Get the data.

        Returns:
            The current data of the node.

        """
        return self.learner.get_data()

    def get_aggregator(self) -> Aggregator:
        """
        Get the node aggregator.

        Returns:
            The node aggregator.

        """
        return self.aggregator

    def get_communication_protocol(self) -> CommunicationProtocol:
        """
        Get the communication protocol.

        Returns:
            The node communication protocol.

        """
        return self.communication_protocol

    def get_learner(self) -> Learner:
        """
        Get the learner.

        Returns:
            The current learner of the node.

        """
        return self.learner

    def get_learning_workflow(self) -> LearningWorkflowModel:
        """
        Get the learning workflow.

        Returns:
            The current learning workflow of the node.

        """
        return self.learning_workflow

    def get_workflow_type(self) -> WorkflowType:
        """
        Get the workflow type.

        Returns:
            The current workflow type of the node.

        """
        return self.workflow_type

    #######################
    #    State Getters    #
    #######################

    def get_local_state(self) -> LocalNodeState:
        """
        Get the local state.

        Returns:
            The current local state of the node.

        """
        return self.local_state

    def get_generator(self) -> random.Random:
        """
        Get the generator.

        Returns:
            The current generator of the node.

        """
        return self.generator

    @property
    def address(self) -> str:
        """The node address."""
        return self.get_local_state().address

    #######################
    #    State Setters    #
    #######################
    def set_local_state(self, state: LocalNodeState) -> None:
        """
        Set the local state.

        Args:
            state: The local state to be set.

        """
        self.local_state = state

    ###############################################
    #         Network Learning Management         #
    ###############################################
    async def run_workflow_loop(self):
        """Continuously checks and steps through the workflow."""
        try:
            while not self.learning_workflow.finished:
                stepped = await self.learning_workflow.next_stage()
                if not stepped:
                    await asyncio.sleep(1)  # Prevent tight loop on no transition
        except asyncio.CancelledError:
            logger.info(self.address, "⛔ Workflow loop cancelled.")
        except Exception as e:
            logger.error(self.address, f"🔥 Unexpected error in workflow loop: {type(e).__name__}: {e}")

    def set_learning_workflow(self,
        workflow: WorkflowType
        ) -> None:
        """
        Set the learning workflow.

        Args:
            workflow: The type of workflow to be used.

        """
        if not self.learning_workflow.waiting_for_learning_start:
            raise NodeRunningException("Cannot set learning workflow while learning is in progress.")

        # Workflow factory
        self.workflow_factory = WorkflowFactoryProducer.get_factory(workflow)

        # Create workflow
        self.learning_workflow, self.learning_machine = self.workflow_factory.create_training_workflow(self)
        self.communication_protocol.add_command(self.workflow_factory.create_commands(self))

        # Set custom model
        model = self.learner.get_P2PFLModel()
        self.learner.set_P2PFLModel(self.workflow_factory.create_model(model))

    @dualmethod
    async def set_start_learning(self,
        rounds: int = 1,
        epochs: int = 1,
        trainset_size: int = 4,
        experiment_name: str = "experiment",
        workflow: WorkflowType = WorkflowType.BASIC,
        ) -> str:
        """
        Start the learning process in the entire network.

        Args:
            rounds: Number of rounds of the learning process.
            epochs: Number of epochs of the learning process.
            trainset_size: Size of the trainset.
            experiment_name: The name of the experiment.

        Raises:
            ZeroRoundsException: If rounds is less than 1.

        """
        # Check is running
        if not self.is_running():
            raise NodeRunningException("Node is not running.")

        if rounds < 1:
            raise ZeroRoundsException("Rounds must be greater than 0.")

        self.set_learning_workflow(workflow)

        experiment_name = f"{experiment_name}-{time.time()}"

        try:
            await self.learning_workflow.start_learning(
                experiment_name=experiment_name,
                rounds=rounds,
                epochs=epochs,
                trainset_size=trainset_size,
                workflow_type=workflow.value,
            )

            # Start automatic workflow loop
            #self._workflow_task = asyncio.create_task(self.run_workflow_loop())

            return experiment_name

        except Exception as e:
            logger.error(self.address, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
            raise

    @dualmethod
    async def set_stop_learning(self) -> None:
        """Stop the learning process in the entire network."""
        if self.local_state.round is not None:
            # send stop msg
            await self.communication_protocol.broadcast_gossip(self.communication_protocol.build_msg(StopLearningCommand.get_name()))
            # stop learning
            await self.__stop_learning()
        else:
            logger.info(self.address, "🛑 No learning in progress to stop.")

    ##################################
    #         Local Learning         #
    ##################################

    async def __stop_learning(self) -> None:
        # TODO: Use the workflow to clean up the learning process
        logger.info(self.address, "🛑 Stopping learning")

        # Communication Protocol
        self.communication_protocol.remove_command(self.workflow_factory.create_commands(self))

        # Learner
        await self.learner.interrupt_fit()
        # State
        self.local_state.clear()
        logger.experiment_finished(self.address)

        # Workflow
        self.learning_machine.remove_model(self.learning_workflow)
        self.learning_workflow = LearningWorkflowModel(self)
