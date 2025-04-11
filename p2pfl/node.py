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

import contextlib
import os
import random
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from transitions import MachineError

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
from p2pfl.stages.workflow_factory import WorkflowFactoryProducer
from p2pfl.stages.workflow_type import WorkflowType
from p2pfl.stages.workflows import TrainingWorkflow

# Disbalbe grpc log (pytorch causes warnings)
if logger.get_level_name(logger.get_level()) != "DEBUG":
    os.environ["GRPC_VERBOSITY"] = "NONE"

from p2pfl.network_state import NetworkState, PeerNodeState

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
        addr: str = "",
        learner: Optional[Learner] = None,
        aggregator: Optional[Aggregator] = None,
        protocol: Optional[CommunicationProtocol] = None,
        simulation: bool = False,
        workflow: WorkflowType = WorkflowType.BASIC,
        **kwargs,
    ) -> None:
        """Initialize a node."""
        # Communication protol
        self.communication_protocol = GrpcCommunicationProtocol() if protocol is None else protocol
        self.addr = self.communication_protocol.set_addr(addr)

        # Workflow
        workflow_factory = WorkflowFactoryProducer.get_factory(workflow)
        self.learning_workflow_class: type[TrainingWorkflow] = workflow_factory.create_training_workflow()
        self.learning_workflow: TrainingWorkflow = None
        commands = workflow_factory.create_commands(self)
        model = workflow_factory.create_model(model)

        self.communication_protocol.add_command(commands)

        # Aggregator
        self.aggregator = FedAvg() if aggregator is None else aggregator
        self.aggregator.set_addr(self.addr)

        # Learner
        if learner is None:  # if no learner, use factory default
            learner = LearnerFactory.create_learner(model)()
        self.learner = try_init_learner_with_ray(learner)
        self.learner.set_addr(self.addr)
        self.learner.set_model(model)
        self.learner.set_data(data)
        self.learner.indicate_aggregator(self.aggregator)

        # State
        self.local_state = LocalNodeState(self.addr)
        self.network_state = NetworkState()


        # Simulation
        self.generator: random.Random = random.Random(Settings.general.SEED)

        self.executor = ThreadPoolExecutor(max_workers=20)


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
        logger.info(self.addr, f"Removing {addr}...")
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
        return self.learning_workflow is not None

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

        self.learning_workflow = self.learning_workflow_class(self)

        # P2PFL Web Services
        logger.register_node(self.addr)
        # Communication Protocol
        await self.communication_protocol.start()
        if wait:
            self.communication_protocol.wait_for_termination()
            logger.info(self.addr, "gRPC terminated.")

    async def stop(self) -> None:
        """
        Stop the node: server and neighbors(gossip and heartbeat).

        Raises:
            NodeRunningException: If the node is not running.

        """
        logger.info(self.addr, "Stopping node...")
        try:
            # Stop server
            await self.communication_protocol.stop()
            # State
            self.local_state.clear()
            # Unregister node
            logger.unregister_node(self.addr)
        except Exception:
            pass

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
        self.learner.set_model(model)

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
        return self.learner.get_model()

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

    def get_network_state(self) -> NetworkState:
        """
        Get the network state.

        Returns:
            The current network state of the node.

        """
        return self.network_state

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
        return self.state.addr

    ###############################################
    #         Network Learning Management         #
    ###############################################
    async def set_start_learning(self, rounds: int = 1, epochs: int = 1, trainset_size: int = 4, experiment_name="experiment") -> str:
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

        experiment_name = f"{experiment_name}-{time.time()}"

        try:
            await self.learning_workflow.send_starting_learning(experiment_name, rounds, epochs, trainset_size)

            return experiment_name

        except MachineError as e:
            logger.debug(self.addr, f"Learning already started: {e}")
        except Exception as e:
            logger.error(self.addr, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
            raise

    async def set_stop_learning(self) -> None:
        """Stop the learning process in the entire network."""
        if self.local_state.round is not None:
            # send stop msg
            await self.communication_protocol.broadcast(self.communication_protocol.build_msg(StopLearningCommand.get_name()))
            # stop learning
            await self.__stop_learning()
        else:
            logger.info(self.addr, "Learning already stopped")

    ##################################
    #         Local Learning         #
    ##################################

    async def __stop_learning(self) -> None:
        logger.info(self.addr, "Stopping learning")

        # Learner
        self.learner.interrupt_fit()
        # Aggregator
        self.aggregator.clear()
        # State
        self.local_state.clear()
        logger.experiment_finished(self.addr)
        # Try to free wait locks
        with contextlib.suppress(Exception):
            self.local_state.wait_votes_ready_lock.release()

        self.learning_workflow = None
