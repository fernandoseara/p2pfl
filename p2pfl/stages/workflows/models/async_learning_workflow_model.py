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

from typing import TYPE_CHECKING

from transitions.extensions.nesting import NestedState

from p2pfl.communication.commands.message.asyDFL.ctx_info_updating_command import (
    IndexInformationUpdatingCommand,
    LossInformationUpdatingCommand,
)
from p2pfl.communication.commands.message.node_initialized_command import NodeInitializedCommand
from p2pfl.communication.commands.message.start_learning_command import StartLearningCommand
from p2pfl.learning.frameworks.custom_model_factory import CustomModelFactory
from p2pfl.learning.frameworks.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.learning.frameworks.tensorflow.custom_models.asydfl_model import AsyDFLKerasP2PFLModel
from p2pfl.learning.frameworks.tensorflow.custom_models.custom_model_factory import KerasCustomModelFactory
from p2pfl.management.logger import logger
from p2pfl.stages.asyDFL.compute_priority_stage import ComputePriorityStage
from p2pfl.stages.asyDFL.gossip_model_stage import GossipModelStage
from p2pfl.stages.asyDFL.select_neighbor_stage import SelectNeighborsStage
from p2pfl.stages.network_state.async_network_state import AsyncNetworkState
from p2pfl.stages.workflows.models.learning_workflow_model import LearningWorkflowModel

NestedState.separator = '↦'

if TYPE_CHECKING:
    from p2pfl.node import Node


class AsyncLearningWorkflowModel(LearningWorkflowModel):
    """Model for the training workflow."""

    def __init__(self, node: Node):
        """Initialize the workflow model."""
        super().__init__(node=node)

        # Set the initial state
        self.network_state: AsyncNetworkState = AsyncNetworkState()

    ###################
    # STATE CALLBACKS #
    ###################
    async def on_enter_starting_training(
        self,
        experiment_name: str,
        rounds: int = 0,
        epochs: int = 0,
        trainset_size: int = 0,
        workflow_type: str = "AsyDFL",
        source: str | None = None
        ):
        """Start the training."""
        logger.info(self.node.address, "⏳ Starting training.")

        local_state = self.node.get_local_state()
        learner = self.node.get_learner()
        communication_protocol = self.node.get_communication_protocol()

        # Set the experiment parameters
        local_state.set_experiment(experiment_name, rounds, epochs, trainset_size)
        learner.set_epochs(local_state.epochs)

        # Comunicate the start of the learning process
        await communication_protocol.broadcast(communication_protocol.build_msg(
            StartLearningCommand.get_name(),
            [local_state.total_rounds,
            self.node.get_learner().get_epochs(),
            local_state.get_experiment().trainset_size,
            local_state.get_experiment().exp_name,
            workflow_type]
            )
        )

        # Initialize asynchronous DFL variables
        logger.info(self.node.address, "Initializing local model and parameters...")
        #learner.get_P2PFLModel().get_model().de_biased_model = learner.get_P2PFLModel().get_model().clone_model()  # χ(0) = ω(0)
        self.tau = 2  # τ

        # Setup learner
        learner.set_steps_per_epoch(1)
        learner.set_epochs(1)

        await self.next_stage()

    async def on_enter_waiting_for_synchronization(self, *args, **kwargs):
        """Wait for the synchronization."""
        logger.info(self.node.address, "⏳ Waiting initialization.")

        communication_protocol = self.node.get_communication_protocol()

        # Communicate Initialization
        await communication_protocol.broadcast(communication_protocol.build_msg(NodeInitializedCommand.get_name()))

        # Set self model initialized
        await self.node.learning_workflow.node_started(
            source=self.node.address,
        )

    async def on_enter_nodes_synchronized(self, *args, **kwargs):
        """Nodes synchronized."""
        communication_protocol = self.node.get_communication_protocol()
        local_state = self.node.get_local_state()
        neighbors = list(communication_protocol.get_neighbors(only_direct=True).keys())

        # Set train set
        local_state.train_set = neighbors + [self.node.address]
        logger.info(self.node.address, f"👥 Peers in train set: {local_state.train_set}")

        self.network_state.set_mixing_weights({neighbor: 1.0 / len(neighbors) if neighbors else 1.0 for neighbor in neighbors})  # pt_j,i(0)

        await self.next_stage()

    async def on_enter_debiasing_model(self, *args, **kwargs):
        """Round initialized."""
        logger.debug(self.node.address, "🤖 Debiasing model.")

        # De-bias the update model (Equation 3)
        self.node.get_learner().get_P2PFLModel().set_push_sum_weight(self.network_state.get_push_sum_weight(self.node.address))

        await self.next_stage()

    async def on_enter_updating_local_model(self, *args, **kwargs):
        """Update the round."""
        logger.info(self.node.address, "🏋️‍♀️ Updating local model...")
        await self.node.get_learner().train_on_batch()
        self.network_state.add_model(self.node.get_learner().get_P2PFLModel(), self.node.address)

        await self.next_stage()

    async def on_enter_sending_training_loss(self, *args, **kwargs):
        """Send the training loss."""
        communication_protocol = self.node.get_communication_protocol()
        local_state = self.node.get_local_state()

        training_loss = self.node.get_learner().get_P2PFLModel().get_last_training_loss()
        self.network_state.add_loss(self.node.address, local_state.round, training_loss)

        logger.info(self.node.address, "📢 Broadcasting loss values.")
        flattened_loss = [str(training_loss)]
        await communication_protocol.broadcast(
            communication_protocol.build_msg(
                LossInformationUpdatingCommand.get_name(),
                flattened_loss,
                round=local_state.round,
            )
        )

        await self.next_stage()

    async def on_enter_gossipping_model(self):
        """Gossip the model."""
        logger.info(self.node.address, "📡 Gossiping model...")
        await GossipModelStage.execute(
            network_state=self.network_state,
            candidates=self.candidates,
            node=self.node,
        )

        await self.next_stage()

    async def on_enter_aggregating(self):
        """Aggregate the models."""
        communication_protocol = self.node.get_communication_protocol()
        local_state = self.node.get_local_state()

        logger.info(self.node.address, "🤝 Aggregating models...")

        push_sum_weight = self.network_state.get_push_sum_weight(self.node.address)

        for neighbor, neighbor_state in self.network_state.get_all_peers().items():
            if neighbor == self.node.address:
                continue

            # Update push-sum weights (Equation 5)
            push_sum_weight += neighbor_state.mixing_weight * neighbor_state.push_sum_weight
            logger.debug(self.node.address, f"📡 {neighbor} push-sum weight updated to {push_sum_weight}")

            # Update index of P2P updating
            self.network_state.update_p2p_updating_idx(neighbor,local_state.round)

            # Send index of local iteration to neighbors
            await communication_protocol.send(
                nei=neighbor,
                msg=communication_protocol.build_msg(
                    IndexInformationUpdatingCommand.get_name(),
                    round=local_state.round,
                )
            )

        # P2P updating of the model
        agg_model = self.node.get_aggregator().aggregate(self.network_state.get_all_models())
        self.node.get_learner().set_P2PFLModel(agg_model)

        await self.next_stage()

    async def on_enter_network_updating_finishing(self):
        """Finish the network updating."""
        logger.info(self.node.address, "🏁 Network updating finished.")

    async def on_final_network_updating(self):
        """Finish the network updating."""
        await self.next_stage()

    async def on_enter_training_round_finishing(self):
        """Finish the round."""
        logger.info(
            self.node.address,
            f"🎉 Round {self.node.get_local_state().round} finished.",
        )

        self.node.get_local_state().increase_round()
        self.network_state.clear_all_models()

    async def on_final_training_round(self):
        """Finish the training round."""
        await self.next_stage()

    async def on_enter_training_finished(self):
        """Finish the training."""
        # Communication Protocol
        self.node.get_communication_protocol().remove_command(self.node.workflow_factory.create_commands(self))

        # Clean state
        self.node.get_local_state().clear()
        self.network_state.clear()

        logger.info(self.node.address, "😋 Training finished!!")

        await self.training_finished()


    ##############
    # CONDITIONS #
    ##############
    def candidate_exists(self, *args, **kwargs):
        """Check if there are candidates."""
        return len(self.candidates) > 0

    def is_all_nodes_started(self, *args, **kwargs):
        """Check if all nodes have started."""
        return len(self.network_state.list_peers()) == (len(self.node.communication_protocol.get_neighbors(only_direct=True)) + 1)

    def is_total_rounds_reached(self, *args, **kwargs):
        """Check if the total rounds have been reached (Line 2)."""
        return self.node.get_local_state().round >= self.node.get_local_state().total_rounds

    def is_iteration_network_updating(self, *args, **kwargs):
        """Check if it is time to update the model with network updating (Line 7)."""
        return self.node.get_local_state().round % self.tau == 0



    ###########################
    # EVENT HANDLER CALLBACKS #
    ###########################
    async def on_enter_waiting_context_update(self, *args, **kwargs):
        """Wait for the context to be updated."""
        logger.info(self.node.address, "⏳ Waiting for the context to be updated.")


    ########################
    # EVENT HANDLER EVENTS #
    ########################
    async def send_network_ready(self, *args, **kwargs):
        """Send the network ready event."""
        logger.info(self.node.address, "✅ Network ready.")
        await self.network_ready()


    #########################
    # EVENT HANDLER SETTERS #
    #########################
    async def create_peer(self, source: str):
        """Update the peer round."""
        self.network_state.add_peer(source)
        logger.debug(self.node.address, f"📡 {source} peer created")

    async def save_started_node(self, source: str):
        """Save the votes."""
        self.network_state.update_round(source, -1)
        logger.debug(self.node.address, f"📦 Node started: {source}")

    async def save_loss_information(self, source: str, round: int, loss: float):
        """Save the loss information."""
        self.network_state.add_loss(source, round, loss)
        logger.debug(self.node.address, f"📡 {source} loss updated to {loss} for round {round}")

    async def save_model(self,
        source: str,
        round: int,
        weights: bytes,
        num_samples: int,
        contributors: list[str]
        ):
        """Initialize model."""
        # Check source
        # Wait and gossip model initialization
        logger.info(self.node.address, "📦 Model received.")

        try:
            model = self.node.get_learner().get_P2PFLModel().build_copy(params=weights, num_samples=num_samples, contributors=contributors)
            self.network_state.add_model(model, source)

        except DecodingParamsError:
            logger.error(self.node.address, "❌ Error decoding parameters.")
        except ModelNotMatchingError:
            logger.error(self.node.address, "❌ Models not matching.")
        except Exception as e:
            logger.error(self.node.address, f"❌ Unknown error adding model: {e}")

    async def save_iteration_index(self,
                            source: str,
                            index: int,
                            ):
        """Save the iteration index."""
        self.network_state.update_round(source, index)
        logger.debug(self.node.address, f"📡 {source} round updated to {index}")

    async def save_push_sum_weight(self, source: str, push_sum_weight: float):
        """Save the push sum weight."""
        self.network_state.update_push_sum_weight(source, push_sum_weight)
        logger.debug(self.node.address, f"📡 {source} push sum weight updated to {push_sum_weight}")





    ########################
    # CANDIDATES CALLBACKS #
    ########################
    async def get_gossip_candidates(self):
        """Get the candidates from the train set to gossip the partial model."""
        # Compute the candidates priorities
        neighbor_priorities = await ComputePriorityStage.execute(
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
