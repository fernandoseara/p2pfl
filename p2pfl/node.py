#
# This file is part of the p2pfl distribution (see https://github.com/pguijas/p2pfl).
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

"""P2PFL Node."""

from __future__ import annotations

import os
import random
import time
import traceback
from typing import TYPE_CHECKING, Any

from p2pfl.communication.commands.message.metrics_command import MetricsCommand
from p2pfl.communication.commands.message.start_learning_command import StartLearningCommand
from p2pfl.communication.commands.message.stop_learning_command import StopLearningCommand
from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
from p2pfl.communication.protocols.protobuff.grpc import GrpcCommunicationProtocol
from p2pfl.exceptions import LearnerRunningException, NodeRunningException, ZeroRoundsException
from p2pfl.learning.aggregators import Aggregator, get_default_aggregator
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.learning.frameworks.learner_factory import LearnerFactory
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.learning.frameworks.ray import try_init_learner_with_ray
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.stages.local_state.node_state import LocalNodeState
from p2pfl.stages.workflow_type import WorkflowType
from p2pfl.stages.workflows.node_workflow import NodeWorkflowModel
from p2pfl.utils.asyncio import sync_or_async

if TYPE_CHECKING:
    from p2pfl.stages.workflows.models.learning_workflow_model import LearningWorkflowModel

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
        learner: Learner | None = None,
        aggregator: Aggregator | None = None,
        protocol: CommunicationProtocol | None = None,
        **kwargs,
    ) -> None:
        """Initialize a node."""
        # Communication protocol
        self.communication_protocol = GrpcCommunicationProtocol() if protocol is None else protocol
        address = self.communication_protocol.set_address(address)

        # self.communication_protocol.add_command(commands)

        # Select default aggregator based on model type if not provided
        if aggregator is None:
            aggregator = get_default_aggregator(model)

        # Validate model-aggregator compatibility early (fail-fast)
        aggregator.validate_models([model])

        # Aggregator
        self.aggregator = aggregator
        self.aggregator.set_address(address)

        # Learner
        if learner is None:  # if no learner, use factory default
            learner = LearnerFactory.create_learner(model)()
        self.learner = try_init_learner_with_ray(learner)
        self.learner.set_address(address)
        self.learner.set_model(model)
        self.learner.set_data(data)
        self.learner.indicate_aggregator(self.aggregator)

        # Simulation
        self.generator: random.Random = random.Random(Settings.general.SEED)

        # Workflow
        self.address: str = address
        self.node_workflow: NodeWorkflowModel = NodeWorkflowModel(self)

        # Commands
        self.communication_protocol.add_command(
            [
                StartLearningCommand(self),
                StopLearningCommand(self),
                MetricsCommand(),
            ]
        )

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
        return self.node_workflow.connect_node(addr)

    def get_neighbors(self, only_direct: bool = False) -> dict[str, Any]:
        """
        Return the neighbors of the node.

        Args:
            only_direct: If True, only the direct neighbors will be returned.

        Returns:
            The list of neighbors.

        """
        return self.communication_protocol.get_neighbors(only_direct)

    def disconnect(self, addr: str, close_code: int) -> None:
        """
        Disconnect a node from another.

        Args:
            addr: The address of the node to disconnect from.
            close_code: The close code for the disconnection.

        """
        self.node_workflow.disconnect_node(addr)

    #######################################
    #   Node Management (servicer loop)   #
    #######################################
    async def start(self, wait: bool = False) -> None:
        """
        Start the node: server and neighbors(gossip and heartbeat).

        Args:
            wait: If True, the function will wait until the server is terminated.

        Raises:
            MachineError: If the node is already running.

        """
        await self.node_workflow.start_node(wait)

    async def stop(self) -> None:
        """
        Stop the node: server and neighbors (gossip and heartbeat).

        Raises:
            MachineError: If the node is not running.

        """
        await self.node_workflow.stop_node()

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
        if self.get_local_state().round is not None:
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
        if self.get_local_state().round is not None:
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
        if self.get_local_state().round is not None:
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

    def get_node_workflow(self) -> NodeWorkflowModel:
        """
        Get the node workflow.

        Returns:
            The current node workflow of the node.

        """
        return self.node_workflow

    def get_learning_workflow(self) -> LearningWorkflowModel:
        """
        Get the learning workflow.

        Returns:
            The current learning workflow of the node.

        Raises:
            NodeRunningException: If the node workflow is not initialized.

        """
        learning_workflow = self.node_workflow.get_learning_workflow()
        if learning_workflow is not None:
            return learning_workflow
        else:
            raise NodeRunningException("Node workflow is not initialized")

    #######################
    #    State Getters    #
    #######################

    def get_local_state(self) -> LocalNodeState:
        """
        Get the local state.

        Returns:
            The current local state of the node.

        """
        return self.node_workflow.get_local_state()

    def get_generator(self) -> random.Random:
        """
        Get the generator.

        Returns:
            The current generator of the node.

        """
        return self.generator

    ###############################################
    #         Network Learning Management         #
    ###############################################

    @sync_or_async
    async def set_start_learning(
        self,
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
            workflow: The workflow type to use.

        Raises:
            ZeroRoundsException: If rounds is less than 1.

        """
        if rounds < 1:
            raise ZeroRoundsException("Rounds must be greater than 0.")

        experiment_name = f"{experiment_name}-{time.time()}"

        try:
            await self.node_workflow.start_learning(
                workflow_type=workflow,
                experiment_name=experiment_name,
                rounds=rounds,
                epochs=epochs,
                trainset_size=trainset_size,
            )

            return experiment_name

        except Exception as e:
            logger.error(self.address, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
            raise

    @sync_or_async
    async def set_stop_learning(self) -> None:
        """Stop the learning process in the entire network."""
        local_state = self.get_local_state()
        if local_state.round is not None:
            # send stop msg
            await self.communication_protocol.broadcast_gossip(self.communication_protocol.build_msg(StopLearningCommand.get_name()))
        else:
            logger.info(self.address, "🛑 No learning in progress to stop.")
