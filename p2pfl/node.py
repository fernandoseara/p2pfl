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

from p2pfl.communication.commands.infrastructure import MetricsCommand, StartLearningCommand, StopLearningCommand
from p2pfl.communication.commands.workflow.workflow_command import WorkflowCommand
from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
from p2pfl.communication.protocols.protobuff.grpc import GrpcCommunicationProtocol
from p2pfl.exceptions import LearnerRunningException, NodeRunningException
from p2pfl.learning.aggregators import Aggregator, get_default_aggregator
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.learning.frameworks.learner_factory import LearnerFactory
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.learning.frameworks.ray import try_init_learner_with_ray
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState, NodeStatus
from p2pfl.settings import Settings
from p2pfl.workflow.engine.experiment import Experiment
from p2pfl.workflow.factory import create_workflow

if TYPE_CHECKING:
    from p2pfl.workflow.engine.workflow import Workflow

# Disable grpc log (pytorch causes warnings)
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
    >>> await node.start()
    >>> await node.connect("127.0.0.1:666")
    >>> await node.set_start_learning(rounds=2, epochs=1)

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

        # Node state
        self.address: str = address
        self._running: bool = False
        self.workflow: Workflow | None = None

        # Commands
        self.communication_protocol.add_command(
            [
                StartLearningCommand(self),
                StopLearningCommand(self),
                MetricsCommand(),
            ]
        )

    #############################
    #  Neighborhood management  #
    #############################

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
        return await self.communication_protocol.connect(addr)

    def get_neighbors(self, only_direct: bool = False) -> dict[str, Any]:
        """
        Return the neighbors of the node.

        Args:
            only_direct: If True, only the direct neighbors will be returned.

        Returns:
            The list of neighbors.

        """
        return self.communication_protocol.get_neighbors(only_direct)

    async def disconnect(self, addr: str) -> None:
        """
        Disconnect a node from another.

        Args:
            addr: The address of the node to disconnect from.

        """
        logger.info(self.address, f"Removing {addr}...")
        await self.communication_protocol.disconnect(addr)

    #######################################
    #   Node Management (servicer loop)   #
    #######################################
    async def start(self, wait: bool = False) -> None:
        """
        Start the node: server and neighbors (gossip and heartbeat).

        Args:
            wait: If True, the function will wait until the server is terminated.

        Raises:
            NodeRunningException: If the node is already running.

        """
        if self._running:
            raise NodeRunningException("Node is already running.")

        logger.info(self.address, "🚀 Starting node.")
        logger.register_node(self.address)
        await self.communication_protocol.start()
        self._running = True

        if wait:
            await self.communication_protocol.wait_for_termination()
            logger.info(self.address, "Node server terminated.")

    async def stop(self) -> None:
        """
        Stop the node: server and neighbors (gossip and heartbeat).

        Raises:
            NodeRunningException: If the node is not running.

        """
        if not self._running:
            raise NodeRunningException("Node is not running.")

        logger.info(self.address, "🛑 Stopping node.")

        try:
            # Stop/clean up workflow if present (active or terminal)
            if self.workflow is not None:
                try:
                    if self.state.is_learning:
                        await self.communication_protocol.broadcast_gossip(
                            self.communication_protocol.build_msg(StopLearningCommand.get_name())
                        )
                    await self._stop_workflow()
                except Exception as e:
                    logger.warning(self.address, f"Error stopping learning during shutdown: {e}")
        finally:
            try:
                # Always stop the server, even if learning shutdown failed
                await self.communication_protocol.stop()
                logger.unregister_node(self.address)
            except Exception as e:
                logger.warning(self.address, f"Error stopping server: {e}")
            self._running = False

    ##############################
    #    Learning Properties     #
    ##############################

    def _check_not_learning(self) -> None:
        """Raise if learning is in progress."""
        if self.state.is_learning:
            raise LearnerRunningException("Cannot modify node while learning is in progress.")

    @property
    def model(self) -> P2PFLModel:
        """Get the current model of the node."""
        return self.learner.get_model()

    @model.setter
    def model(self, model: P2PFLModel) -> None:
        """Set the model."""
        self._check_not_learning()
        self.learner.set_model(model)

    @property
    def data(self) -> P2PFLDataset:
        """Get the current data of the node."""
        return self.learner.get_data()

    @data.setter
    def data(self, data: P2PFLDataset) -> None:
        """Set the data."""
        self._check_not_learning()
        self.learner.set_data(data)

    @property
    def state(self) -> NodeState:
        """Get the unified lifecycle state of the node."""
        if not self._running:
            return NodeState.OFFLINE
        if self.workflow is None:
            return NodeState.IDLE
        return NodeState.from_workflow_status(self.workflow.status)

    @property
    def status(self) -> NodeStatus:
        """Get a complete snapshot of the node's status."""
        wf = self.workflow
        return NodeStatus(
            address=self.address,
            state=self.state,
            num_neighbors=len(self.communication_protocol.get_neighbors(only_direct=False)),
            experiment=wf.experiment if wf is not None else None,
            error=str(wf.error) if wf is not None and wf.error is not None else None,
            current_stage_name=wf.current_stage_name if wf is not None else None,
        )

    ###############################################
    #         Network Learning Management         #
    ###############################################

    async def set_start_learning(
        self,
        rounds: int = 1,
        epochs: int = 1,
        experiment_name: str = "experiment",
        workflow: str = "basic",
        **kwargs: Any,
    ) -> str:
        """
        Start the learning process in the entire network.

        Args:
            rounds: Number of rounds of the learning process.
            epochs: Number of epochs of the learning process.
            experiment_name: The name of the experiment.
            workflow: The workflow type to use.
            **kwargs: Workflow-specific parameters (e.g. trainset_size for BasicDFL).

        Raises:
            ZeroRoundsException: If rounds is less than 1.
            NodeRunningException: If learning is already in progress.

        """
        experiment_name = f"{experiment_name}-{time.time()}"

        try:
            # Validate by constructing Experiment before broadcasting
            experiment = Experiment.create(
                exp_name=experiment_name,
                total_rounds=rounds,
                epochs_per_round=epochs,
                workflow=workflow,
                is_initiator=True,
                **kwargs,
            )

            # Broadcast start learning command to the network
            await self.communication_protocol.broadcast_gossip(
                self.communication_protocol.build_msg(
                    StartLearningCommand.get_name(),
                    [rounds, epochs, experiment_name, workflow, kwargs],
                )
            )

            await self._start_learning_workflow(workflow, experiment, **kwargs)

            return experiment_name

        except Exception as e:
            logger.error(self.address, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
            raise

    async def _start_learning_workflow(
        self,
        workflow_name: str,
        experiment: Experiment,
        **kwargs: Any,
    ) -> None:
        """
        Start the learning workflow internally.

        Args:
            workflow_name: The registered workflow name (e.g. ``"basic"``, ``"async"``).
            experiment: A fully constructed Experiment describing this run.
            **kwargs: Workflow-specific parameters forwarded to workflow.start().

        Raises:
            NodeRunningException: If learning is already in progress.

        """
        if self.workflow is not None:
            if not self.workflow.status.is_terminal:
                raise NodeRunningException("Learning is already in progress.")
            # Clean up completed/failed/cancelled workflow before starting new one
            self._unregister_workflow_commands()
            self.workflow = None

        logger.info(self.address, f"⏳ Setting environment for learning. Learning type: {workflow_name}")

        # Create workflow using factory
        self.workflow = create_workflow(workflow_name)

        # Register workflow message handlers as communication commands
        self._register_workflow_commands()

        try:
            await self.workflow.start(
                experiment,
                address=self.address,
                learner=self.learner,
                aggregator=self.aggregator,
                cp=self.communication_protocol,
                generator=self.generator,
                **kwargs,
            )
        except Exception:
            self._unregister_workflow_commands()
            self.workflow = None
            raise

    async def set_stop_learning(self) -> None:
        """Stop the learning process in the entire network."""
        if self.workflow is None:
            logger.info(self.address, "🛑 No learning in progress to stop.")
            return
        if self.state.is_learning:
            await self.communication_protocol.broadcast_gossip(self.communication_protocol.build_msg(StopLearningCommand.get_name()))
        await self._stop_workflow()

    async def _stop_workflow(self) -> None:
        """Stop the current workflow and clean up commands. Safe to call multiple times."""
        if self.workflow is not None:
            await self.workflow.stop()
            self._unregister_workflow_commands()
            self.workflow = None

    def _register_workflow_commands(self) -> None:
        """Register workflow message handlers as communication commands."""
        assert self.workflow is not None
        cmds = [WorkflowCommand(self, name) for name in self.workflow.get_messages()]
        if cmds:
            self.communication_protocol.add_command(cmds)

    def _unregister_workflow_commands(self) -> None:
        """Remove workflow message commands from the communication protocol."""
        if self.workflow is None:
            return
        for cmd_name in self.workflow.get_messages():
            self.communication_protocol.remove_command(cmd_name)
