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
import contextlib
from typing import TYPE_CHECKING

from p2pfl.communication.commands.message.asyDFL.ctx_info_updating_command import (
    IndexInformationUpdatingCommand,
    LossInformationUpdatingCommand,
)
from p2pfl.communication.commands.message.node_initialized_command import NodeInitializedCommand
from p2pfl.communication.commands.message.start_learning_command import StartLearningCommand
from p2pfl.learning.frameworks.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.management.logger import logger
from p2pfl.stages.asyDFL.compute_priority_stage import ComputePriorityStage
from p2pfl.stages.asyDFL.gossip_model_stage import GossipModelStage
from p2pfl.stages.asyDFL.select_neighbor_stage import SelectNeighborsStage
from p2pfl.stages.base_node.evaluate_stage import EvaluateStage
from p2pfl.stages.workflows.models.learning_workflow_model import LearningWorkflowModel
from p2pfl.stages.workflows.node_workflow import WorkflowMachineManager
from p2pfl.utils.pytransitions import StateAdapter, TransitionAdapter

if TYPE_CHECKING:
    from p2pfl.node import Node
    from p2pfl.stages.local_state.async_node_state import AsyncLocalNodeState
    from p2pfl.stages.network_state.async_network_state import AsyncNetworkState


# Define states and transitions
def get_states() -> list[dict]:
    """Define the states for the workflow."""
    states = [
        # Training workflow states
        StateAdapter(name="waitingSetup"),
        StateAdapter(name="startingTraining"),
        StateAdapter(name="waitingForSynchronization"),
        StateAdapter(name="nodesSynchronized"),
        StateAdapter(
            name="trainingRound",
            initial="debiasingModel",
            children=[
                StateAdapter(name="debiasingModel"),
                StateAdapter(name="updatingLocalModel"),
                StateAdapter(name="sendingTrainingLoss"),
                StateAdapter(
                    name="networkUpdating",
                    initial="gossipingModel",
                    children=[
                        StateAdapter(name="gossipingModel"),
                        StateAdapter(name="aggregating"),
                        StateAdapter(name="networkUpdatingFinishing", final=True),
                    ],
                ),
                StateAdapter(name="roundFinishing", final=True),
            ],
        ),
        StateAdapter(name="trainingFinished", final=True),
    ]
    return [state.to_dict() for state in states]


def get_transitions() -> list[dict]:
    """Define the transitions for the workflow."""
    transitions = [
        # Training workflow transitions
        # Setup & initial synchronization
        TransitionAdapter(trigger="setup", source="waitingSetup", dest="startingTraining"),
        TransitionAdapter(trigger="next_stage", source="startingTraining", dest="waitingForSynchronization"),
        TransitionAdapter(trigger="network_ready", source="waitingForSynchronization", dest="nodesSynchronized"),
        TransitionAdapter(trigger="next_stage", source="nodesSynchronized", dest="trainingRound"),
        # Debiasing & updating
        TransitionAdapter(trigger="next_stage", source="trainingRound_debiasingModel", dest="trainingRound_updatingLocalModel"),
        TransitionAdapter(trigger="next_stage", source="trainingRound_updatingLocalModel", dest="trainingRound_sendingTrainingLoss"),
        # Check if is it time to update the model with network updating
        TransitionAdapter(
            trigger="next_stage",
            source="trainingRound_sendingTrainingLoss",
            dest="trainingRound_networkUpdating",
            conditions="check_iteration_network_updating",
            before="get_gossip_candidates",
        ),
        TransitionAdapter(trigger="next_stage", source="trainingRound_sendingTrainingLoss", dest="trainingRound_roundFinishing"),
        # Network updating
        TransitionAdapter(
            trigger="next_stage", source="trainingRound_networkUpdating_gossipingModel", dest="trainingRound_networkUpdating_aggregating"
        ),
        TransitionAdapter(
            trigger="next_stage",
            source="trainingRound_networkUpdating_aggregating",
            dest="trainingRound_networkUpdating_networkUpdatingFinishing",
        ),
        TransitionAdapter(trigger="next_stage", source="trainingRound_networkUpdating", dest="trainingRound_roundFinishing"),
        # Loop
        TransitionAdapter(trigger="next_stage", source="trainingRound", dest="trainingFinished", conditions="check_total_rounds_reached"),
        TransitionAdapter(trigger="next_stage", source="trainingRound", dest="nodesSynchronized"),
        # Stop training
        TransitionAdapter(trigger="stop_learning", source="*", dest="trainingFinished", before="stop"),
        # Event handler transitions
        TransitionAdapter(
            trigger="node_started",
            source=["waitingSetup", "startingTraining", "waitingForSynchronization"],
            dest=None,
            prepare="create_peer",
            conditions="is_all_nodes_started",
            after="send_network_ready",
        ),
        TransitionAdapter(trigger="loss_information_received", source="*", dest=None, prepare="save_loss_information"),
        TransitionAdapter(trigger="iteration_index_received", source="*", dest=None, prepare="save_iteration_index"),
        TransitionAdapter(trigger="model_received", source="*", dest=None, prepare="save_model"),
        TransitionAdapter(trigger="push_sum_weight_received", source="*", dest=None, prepare="save_push_sum_weight"),
    ]
    return [transition.to_dict() for transition in transitions]


class AsyncLearningWorkflowModel(LearningWorkflowModel):
    """Model for the training workflow."""

    def __init__(self, node: Node, local_state: AsyncLocalNodeState, network_state: AsyncNetworkState) -> None:
        """Initialize the workflow model."""
        self.network_state: AsyncNetworkState = network_state
        self.local_state: AsyncLocalNodeState = local_state

        super().__init__(
            node=node,
        )

    async def is_finished(self) -> bool:
        """Check if the workflow has finished."""
        return self.state == "trainingFinished"

    ########################################
    # EVENTS (Overridden by pytransitions) #
    ########################################

    # Event handler events
    async def node_started(self, source: str) -> bool:
        """Handle the node started event."""
        raise RuntimeError("Should be overridden!")

    async def loss_information_received(self, source: str, round: int, loss: float) -> bool:
        """Handle the loss information received event."""
        raise RuntimeError("Should be overridden!")

    async def iteration_index_received(self, source: str, index: int) -> bool:
        """Handle the iteration index received event."""
        raise RuntimeError("Should be overridden!")

    async def model_received(self, source: str, round: int, weights: bytes, num_samples: int, contributors: list[str]) -> bool:
        """Handle the model received event."""
        raise RuntimeError("Should be overridden!")

    async def push_sum_weight_received(self, source: str, push_sum_weight: float) -> bool:
        """Handle the push sum weight received event."""
        raise RuntimeError("Should be overridden!")

    async def training_finished(self) -> bool:
        """Handle the training finished event."""
        raise RuntimeError("Should be overridden!")

    # Ready events
    async def network_ready(self) -> bool:
        """Handle the network ready event."""
        raise RuntimeError("Should be overridden!")

    ###################
    # STATE CALLBACKS #
    ###################
    async def on_enter_startingTraining(
        self, is_initiator: bool, experiment_name: str, rounds: int = 0, epochs: int = 0, trainset_size: int = 0, source: str | None = None
    ):
        """Start the training."""
        logger.info(self.node.address, "⏳ Starting training.")

        learner = self.node.get_learner()
        communication_protocol = self.node.get_communication_protocol()

        # Set the experiment parameters
        self.local_state.set_experiment(experiment_name, rounds, epochs, trainset_size)
        learner.set_epochs(epochs)

        experiment = self.local_state.get_experiment()
        if experiment is None:
            raise ValueError("Experiment not initialized")
        exp_name = experiment.exp_name

        # Comunicate the start of the learning process
        await communication_protocol.broadcast(
            communication_protocol.build_msg(
                StartLearningCommand.get_name(),
                [
                    self.local_state.total_rounds,
                    self.node.get_learner().get_epochs(),
                    trainset_size,
                    exp_name,
                    self.node.get_node_workflow().get_workflow_type().value,
                ],
            )
        )

        # Initialize asynchronous DFL variables
        logger.info(self.node.address, "Initializing local model and parameters...")
        self.tau = 2  # τ

        # await self.next_stage()

    async def on_enter_waitingForSynchronization(self, *args, **kwargs):
        """Wait for the synchronization."""
        logger.info(self.node.address, "⏳ Waiting initialization.")

        communication_protocol = self.node.get_communication_protocol()

        await EvaluateStage.execute(
            node=self.node,
        )

        # Communicate Initialization
        await communication_protocol.broadcast(communication_protocol.build_msg(NodeInitializedCommand.get_name()))

        # Set self model initialized
        await self.node_started(
            source=self.node.address,
        )

    async def on_enter_nodesSynchronized(self, *args, **kwargs):
        """Nodes synchronized."""
        communication_protocol = self.node.get_communication_protocol()
        neighbors = list(communication_protocol.get_neighbors(only_direct=True).keys())

        # Set train set
        self.local_state.train_set = neighbors + [self.node.address]
        logger.info(self.node.address, f"👥 Peers in train set: {self.local_state.train_set}")

        self.network_state.set_mixing_weights({neighbor: 1.0 / len(neighbors) if neighbors else 1.0 for neighbor in neighbors})  # pt_j,i(0)

        # await self.next_stage()

    async def on_enter_trainingRound_debiasingModel(self, *args, **kwargs):
        """Round initialized."""
        logger.debug(self.node.address, "🤖 Debiasing model.")

        # De-bias the update model (Equation 3)
        model = self.node.get_learner().get_model()
        model.set_push_sum_weight(self.network_state.get_push_sum_weight(self.node.address))

        # await self.continue_training_round()

    async def on_enter_trainingRound_updatingLocalModel(self, *args, **kwargs):
        """Update the round."""
        logger.info(self.node.address, "🏋️‍♀️ Updating local model...")
        await self.node.get_learner().train_on_batch()
        self.network_state.add_model(self.node.get_learner().get_model(), self.node.address)

        # await self.continue_training_round()

    async def on_enter_trainingRound_sendingTrainingLoss(self, *args, **kwargs):
        """Send the training loss."""
        communication_protocol = self.node.get_communication_protocol()

        training_loss = self.node.get_learner().get_model().last_training_loss
        self.network_state.add_loss(self.node.address, self.local_state.round, training_loss)

        logger.info(self.node.address, "📢 Broadcasting loss values.")
        flattened_loss = [str(training_loss)]
        with contextlib.suppress(Exception):
            await communication_protocol.broadcast(
                communication_protocol.build_msg(
                    LossInformationUpdatingCommand.get_name(),
                    flattened_loss,
                    round=self.local_state.round,
                )
            )

        # await self.continue_training_round()

    async def on_enter_trainingRound_networkUpdating_gossipingModel(self, *args, **kwargs):
        """Gossip the model."""
        logger.info(self.node.address, "📡 Gossiping model...")
        await GossipModelStage.execute(
            network_state=self.network_state,
            candidates=self.candidates,
            node=self.node,
        )

        # await self.continue_network_updating()

    async def on_enter_trainingRound_networkUpdating_aggregating(self):
        """Aggregate the models."""
        communication_protocol = self.node.get_communication_protocol()

        logger.info(self.node.address, "🤝 Aggregating models...")

        push_sum_weight = self.network_state.get_push_sum_weight(self.node.address)

        for neighbor, neighbor_state in self.network_state.get_all_peers().items():
            if neighbor == self.node.address:
                continue

            # Update push-sum weights (Equation 5)
            push_sum_weight += neighbor_state.mixing_weight * neighbor_state.push_sum_weight
            logger.debug(self.node.address, f"📡 {neighbor} push-sum weight updated to {push_sum_weight}")

            # Update index of P2P updating
            self.network_state.update_p2p_updating_idx(neighbor, self.local_state.round)

            # Send index of local iteration to neighbors
            with contextlib.suppress(Exception):
                await communication_protocol.send(
                    nei=neighbor,
                    msg=communication_protocol.build_msg(
                        IndexInformationUpdatingCommand.get_name(),
                        round=self.local_state.round,
                    ),
                )

        # P2P updating of the model
        agg_model = self.node.get_aggregator().aggregate(self.network_state.get_all_models())
        self.node.get_learner().set_model(agg_model)

        # Evaluate the model
        await EvaluateStage.execute(
            node=self.node,
        )

        # await self.continue_network_updating()

    async def on_enter_trainingRound_networkUpdating_networkUpdatingFinishing(self):
        """Finish the network updating."""
        logger.info(self.node.address, "🏁 Network updating finished.")

    # async def on_final_network_updating(self):
    #     """Finish the network updating."""
    #     self._fire_next_stage("continue_training_round")

    async def on_enter_trainingRound_roundFinishing(self):
        """Finish the round."""
        logger.info(
            self.node.address,
            f"🎉 Round {self.local_state.round} finished.",
        )

        self.local_state.increase_iteration()
        self.network_state.clear_all_models()

    # async def on_final_training_round(self):
    #     """Finish the training round."""
    #     await self.next_stage()

    async def on_enter_trainingFinished(self):
        """Finish the training."""
        await EvaluateStage.execute(
            node=self.node,
        )

        # Communication Protocol
        # self.node.get_communication_protocol().remove_command(self.node.get_node_workflow().get_commands())

        # Clean state
        self.local_state.clear()
        self.network_state.clear()

        logger.info(self.node.address, "😋 Training finished!!")

    ##############
    # CONDITIONS #
    ##############
    def check_total_rounds_reached(self, *args, **kwargs):
        """Check if the total rounds have been reached (Line 2)."""
        return self.local_state.round >= self.local_state.total_rounds

    def check_iteration_network_updating(self, *args, **kwargs):
        """Check if it is time to update the model with network updating (Line 7)."""
        return self.local_state.round % self.tau == 0

    ########################
    # CANDIDATES CALLBACKS #
    ########################
    async def get_gossip_candidates(self):
        """Get the candidates from the train set to gossip the partial model."""
        # Compute the candidates priorities
        neighbor_priorities = await ComputePriorityStage.execute(  # TODO: ESTO NO TIENE MUCHO SENTIDO... PARA ESO SE USA UNA FUNCION ...
            network_state=self.network_state,
            node=self.node,
        )
        logger.info(self.node.address, f"👥 Neighbor priorities: {neighbor_priorities}")

        # Select the candidates
        self.candidates = await SelectNeighborsStage.execute(
            neighbor_priorities=neighbor_priorities,
            node=self.node,
        )

        logger.info(self.node.address, f"🧾 Selected neighbors: {self.candidates}")

    #########################
    # EVENT HANDLER SETTERS #
    #########################
    async def create_peer(self, source: str):
        """Update the peer round."""
        try:
            self.network_state.add_peer(source)
            logger.debug(self.node.address, f"📡 {source} peer created")
        except Exception as e:
            logger.error(self.node.address, f"❌ Error creating peer {source}: {e}")

    async def save_started_node(self, source: str):
        """Save the votes."""
        try:
            self.network_state.update_round(source, -1)
            logger.debug(self.node.address, f"📦 Node started: {source}")
        except Exception as e:
            logger.error(self.node.address, f"❌ Error saving started node {source}: {e}")

    async def save_loss_information(self, source: str, round: int, loss: float):
        """Save the loss information."""
        try:
            self.network_state.add_loss(source, round, loss)
            logger.debug(self.node.address, f"📡 {source} loss updated to {loss} for round {round}")
        except Exception as e:
            logger.error(self.node.address, f"❌ Error saving loss information from {source} for round {round}: {e}")

    async def save_model(self, source: str, round: int, weights: bytes, num_samples: int, contributors: list[str]):
        """Initialize model."""
        # Check source
        # Wait and gossip model initialization
        logger.info(self.node.address, "📦 Model received.")

        try:
            model = self.node.get_learner().get_model().build_copy(params=weights, num_samples=num_samples, contributors=contributors)
            self.network_state.add_model(model, source)

        except DecodingParamsError:
            logger.error(self.node.address, "❌ Error decoding parameters.")
        except ModelNotMatchingError:
            logger.error(self.node.address, "❌ Models not matching.")
        except Exception as e:
            logger.error(self.node.address, f"❌ Unknown error adding model: {e}")

    async def save_iteration_index(
        self,
        source: str,
        index: int,
    ):
        """Save the iteration index."""
        try:
            self.network_state.update_round(source, index)
            logger.debug(self.node.address, f"📡 {source} round updated to {index}")
        except Exception as e:
            logger.error(self.node.address, f"❌ Error saving iteration index from {source}: {e}")

    async def save_push_sum_weight(self, source: str, push_sum_weight: float):
        """Save the push sum weight."""
        try:
            self.network_state.update_push_sum_weight(source, push_sum_weight)
            logger.debug(self.node.address, f"📡 {source} push sum weight updated to {push_sum_weight}")
        except Exception as e:
            logger.error(self.node.address, f"❌ Error saving push sum weight from {source}: {e}")

    ##############
    # CONDITIONS #
    ##############
    def is_all_nodes_started(self, *args, **kwargs):
        """Check if all nodes have started."""
        return len(self.network_state.list_peers()) == (len(self.node.communication_protocol.get_neighbors(only_direct=True)) + 1)

    ########################
    # EVENT HANDLER EVENTS #
    ########################
    async def send_network_ready(self, *args, **kwargs):
        """Send the network ready event."""
        logger.info(self.node.address, "✅ Network ready.")
        await self.network_ready()

    #############
    # INTERRUPT #
    #############
    async def stop(self) -> None:
        """Stop the learning workflow."""
        logger.info(self.node.address, "🛑 Stopping learning")

        # Learner
        await self.node.get_learner().interrupt_fit()
        # State
        self.local_state.clear()
        logger.experiment_finished(self.node.address)

    async def interrupt(self) -> None:
        """Interrupt the workflow."""
        await asyncio.sleep(1)
        machine = WorkflowMachineManager().get_machine()
        if machine is not None:
            for task in machine.async_tasks[id(self)]:
                task.cancel()
            machine._transition_queue_dict[id(self)].clear()

        await self.stop()
