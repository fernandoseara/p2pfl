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
from typing import TYPE_CHECKING

from transitions.extensions.nesting import NestedState

from p2pfl.communication.commands.message.node_initialized_command import NodeInitializedCommand
from p2pfl.communication.commands.message.peer_round_updated_command import PeerRoundUpdatedCommand
from p2pfl.communication.commands.message.start_learning_command import StartLearningCommand
from p2pfl.learning.frameworks.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger
from p2pfl.stages.base_node import (
    AggregatingVoteTrainSetStage,
    EvaluateStage,
    GossipFullModelStage,
    GossipPartialModelStage,
    StartLearningStage,
    TrainingFinishedStage,
    TrainStage,
    UpdateRoundStage,
    VoteTrainSetStage,
)

NestedState.separator = '↦'

if TYPE_CHECKING:
    from p2pfl.node import Node


class LearningWorkflowModel:
    """Model for the training workflow."""

    def __init__(self, node: Node):
        """Initialize the workflow model."""
        self.node = node
        self.candidates = []

        self.event_log = []
        self.state_log = []

    @property
    def finished(self) -> bool:
        """
        Check if the workflow is finished.

        Returns:
            bool: True if the workflow is finished, False otherwise.

        """
        return self.is_learning_finished()

    async def set_model_initialized(self, *args, **kwargs):
        """Set the model initialized."""
        # Set the model initialized
        self.node.get_learner().get_model().set_round(0)


    ###################
    # STATE CALLBACKS #
    ###################
    async def on_enter_starting_training(self,
                                         experiment_name: str,
                                         rounds: int =0,
                                         epochs: int=0,
                                         trainset_size: int=0,
                                         source: str | None = None):
        """Start the training."""
        logger.info(self.node.address, "⏳ Starting training.")
        await StartLearningStage.execute(
            experiment_name=experiment_name,
            rounds=rounds,
            epochs=epochs,
            trainset_size=trainset_size,
            node=self.node,
        )

        await self.next_stage()

    async def on_enter_waiting_for_synchronization(self):
        """Wait for the synchronization."""
        communication_protocol = self.node.get_communication_protocol()
        local_state = self.node.get_local_state()

        # Wait and gossip model initialization
        logger.info(self.node.address, "⏳ Waiting initialization.")

        # Communicate Initialization
        try:
            await communication_protocol.broadcast(communication_protocol.build_msg(StartLearningCommand.get_name(),
                                                                                    [local_state.total_rounds,
                                                                                    self.node.get_learner().get_epochs(),
                                                                                    local_state.get_experiment().trainset_size,
                                                                                    local_state.get_experiment().exp_name]
                                                                                    ))

            await communication_protocol.broadcast(communication_protocol.build_msg(NodeInitializedCommand.get_name()))

            # Set self model initialized
            await self.node.learning_workflow.node_started(
                source=self.node.address,
            )
        except Exception as e:
            logger.debug(self.node.address, f"Error broadcasting start learning command: {e}")

    async def on_enter_nodes_synchronized(self, *args, **kwargs):
        """All nodes are synchronized."""
        logger.debug(self.node.address, "🤝 All nodes synchronized.")
        await self.next_stage()

    async def on_enter_round_initialized(self, *args, **kwargs):
        """Round initialized."""
        logger.debug(self.node.address, "🤖 Round initialized.")
        await self.next_stage()

    async def on_enter_updating_round(self, *args, **kwargs):
        """Update the round."""
        await UpdateRoundStage.execute(
            node=self.node,
        )

        await self.node.get_communication_protocol().broadcast(
            self.node.get_communication_protocol().build_msg(PeerRoundUpdatedCommand.get_name(),
                                                    round=self.node.get_local_state().get_experiment().round))

        await self.node.learning_workflow.peer_round_updated(
            self.node.address, self.node.get_local_state().round
        )

        await self.continue_p2p_round_initialization()

    async def on_enter_gossipping_full_model(self):
        """Gossip the model."""
        await GossipFullModelStage.execute(
            candidates=self.candidates,
            node=self.node,
        )

        await self.continue_p2p_round_initialization()

    async def on_enter_starting_voting(self, *args, **kwargs):
        """Set the model initialized."""
        await self.continue_p2p_voting()

    async def on_enter_voting(self):
        """Vote for the train set."""
        logger.info(self.node.address, "⏳ Voting for the train set.")
        await VoteTrainSetStage.execute(
            node=self.node,
        )

        await self.continue_p2p_voting()

    async def on_enter_voting_finished(self, *args, **kwargs):
        """Finish the voting."""
        logger.info(self.node.address, "🤖 Voting finished.")

    async def on_final_p2p_voting(self, *args, **kwargs):
        """Finish the voting."""
        await AggregatingVoteTrainSetStage.execute(
            node=self.node,
        )

        await self.next_stage()

    async def on_enter_evaluating(self):
        """Evaluate the model."""
        await EvaluateStage.execute(
            node=self.node,
        )
        await self.continue_p2p_learning()

    async def on_enter_training(self):
        """Train the model."""
        await TrainStage.execute(
            node=self.node,
        )
        await self.continue_p2p_learning()

    async def on_enter_gossipping_partial_aggregation(self):
        """Gossip the partial model."""
        await GossipPartialModelStage.execute(
            candidates=self.candidates,
            node=self.node,
        )
        await self.continue_p2p_learning()

    async def on_enter_aggregating(self):
        """Aggregate the models."""
        # Set aggregated model
        agg_model = self.node.get_aggregator().aggregate(self.node.get_network_state().get_all_models())
        self.node.get_learner().set_model(agg_model)

        await self.continue_p2p_learning()

    async def on_enter_aggregation_finished(self):
        """Finish the aggregation."""
        logger.info(self.node.address, "🤖 Aggregation finished.")

    async def on_final_p2p_learning(self):
        """Finish the training."""
        await self.next_stage()

    async def on_enter_round_finished(self):
        """Finish the round."""
        logger.info(
            self.node.address,
            f"🎉 Round {self.node.get_local_state().round} finished.",
        )
        await self.next_stage()

    async def on_enter_training_finished(self):
        """Finish the training."""
        await TrainingFinishedStage.execute(
            node=self.node,
        )

        logger.info(self.node.address, "😋 Training finished!!")

        await self.training_finished()

    async def on_final_learning(self):
        """Finish the learning workflow."""
        await self.learning_finished()

    ##############
    # CONDITIONS #
    ##############
    def is_model_initialized(self, *args, **kwargs):
        """Check if the model has been initialized."""
        learner_round = self.node.get_learner().get_model().get_round()
        return learner_round == self.node.get_local_state().round

    def is_model_valid(self,
                        source: str,
                        round: int,
                        weights: bytes):
        """Check if the model has been initialized."""
        return self.node.get_local_state().round+1 == round

    def is_all_nodes_started(self, *args, **kwargs):
        """Check if all nodes have started."""
        return len(self.node.get_network_state().list_peers()) == (len(self.node.communication_protocol.get_neighbors(only_direct=True)) + 1)

    def is_all_models_initialized(self, *args, **kwargs):
        """Check if all models have been initialized."""
        rounds = self.node.get_network_state().get_all_rounds()
        return sum(1 for value in rounds.values() if value == self.node.get_local_state().round) == len(rounds)

    def candidate_exists(self, *args, **kwargs):
        """Check if there are candidates."""
        return len(self.candidates) > 0

    def is_all_votes_received(self, *args, **kwargs):
        """Check if all votes from neis have been received."""
        communication_protocol = self.node.get_communication_protocol()
        # Get all votes
        nc_votes = {
                k: v
                for k, v in self.node.get_network_state().get_all_votes().items()
                if k in list(communication_protocol.get_neighbors(only_direct=False)) or k == self.node.address
            }

        #needed_votes = set(list(communication_protocol.get_neighbors(only_direct=False)) + [self.node.address])

        #return needed_votes == set(nc_votes.keys())

        # Check if none of the needed peers votes are empty
        return all(v for v in nc_votes.values())

    def in_train_set(self, *args, **kwargs):
        """Check if the node is in the train set."""
        return self.node.address in self.node.get_local_state().train_set

    def is_total_rounds_reached(self, *args, **kwargs):
        """Check if the total rounds have been reached."""
        return self.node.get_local_state().round >= self.node.get_local_state().total_rounds

    def is_all_models_received(self, *args, **kwargs):
        """Check if all models have been received."""
        return len(self.node.get_local_state().train_set) == len(self.node.get_network_state().get_all_models())



    ###########################
    # EVENT HANDLER CALLBACKS #
    ###########################
    async def on_enter_waiting_network_start(self, *args, **kwargs):
        """Wait for the node to start."""
        logger.info(self.node.address, "⏳ Waiting for the node to start.")

    async def on_enter_waiting_vote(self, *args, **kwargs):
        """Wait for the vote."""
        logger.info(self.node.address, "⏳ Waiting for votes.")

    async def on_enter_waiting_partial_model(self, *args, **kwargs):
        """Wait for the partial model."""
        logger.info(self.node.address, "⏳ Waiting for partial models.")

    async def on_enter_waiting_full_model(self, *args, **kwargs):
        """Wait for the full model."""
        logger.info(self.node.address, "⏳ Waiting for the full model.")

    async def on_enter_waiting_round_update(self, *args, **kwargs):
        """Wait for the round update."""
        logger.info(self.node.address, "⏳ Waiting for peer round updates.")

    ########################
    # EVENT HANDLER EVENTS #
    ########################
    async def send_network_ready(self, *args, **kwargs):
        """Send the network ready event."""
        await self.network_ready()
        logger.info(self.node.address, "✅ Network ready.")

    async def send_models_ready(self, *args, **kwargs):
        """Send the models updated event."""
        await self.models_ready()
        logger.info(self.node.address, "✅ Models ready.")

    async def send_votes_ready(self, *args, **kwargs):
        """Send the votes ready event."""
        await self.votes_ready()
        logger.info(self.node.address, "✅ Votes ready.")

    async def send_aggregation_ready(self, *args, **kwargs):
        """Send the aggregation ready event."""
        await self.aggregation_ready()
        logger.info(self.node.address, "✅ Aggregation ready.")

    async def send_full_model_ready(self, *args, **kwargs):
        """Send the full model ready event."""
        await self.full_model_ready()
        logger.info(self.node.address, "✅ Full model ready.")

    async def send_peers_ready(self, *args, **kwargs):
        """Send the peers ready event."""
        await self.peers_ready()
        logger.info(self.node.address, "✅ Peers ready.")


    #########################
    # EVENT HANDLER SETTERS #
    #########################
    async def save_started_node(self, source: str):
        """Save the votes."""
        self.node.get_network_state().update_round(source, -1)

    async def save_votes(self, source: str, tmp_votes: list[tuple[str, int]]):
        """Save the votes."""
        for train_set_id, vote in tmp_votes:
            self.node.get_network_state().add_vote(source, train_set_id, vote)

    async def save_aggregation(self, model: P2PFLModel, source: str):
        """Save the aggregation."""
        self.node.get_network_state().add_model(model, source)

    async def create_peer(self, source: str):
        """Update the peer round."""
        self.node.get_network_state().add_peer(source)
        logger.debug(self.node.address, f"📡 {source} peer created")

    async def update_peer_round(self, source: str, round: int):
        """Update the peer round."""
        self.node.get_network_state().update_round(source, round)
        logger.debug(self.node.address, f"📡 {source} round updated to {round}")


    async def save_full_model(self,
                            source: str,
                            round: int,
                            weights: bytes):
        """Initialize model."""
        # Check source
        # Wait and gossip model initialization
        logger.info(self.node.address, "⏳ Waiting initialization.")

        try:
            # Set new weights
            self.node.get_learner().set_model(weights)

            # Set self model round
            #self.node.get_network_state().update_round(self.node.address, round)

            logger.info(self.node.address, "🤖 Model Weights Initialized")

        except DecodingParamsError:
            logger.error(self.node.address, "Error decoding parameters.")

        except ModelNotMatchingError:
            logger.error(self.node.address, "Models not matching.")

        except Exception as e:
            logger.error(self.node.address, f"Unknown error adding model: {e}")

    async def save_peer_round_updated(self,
                            source: str,
                            round: int,
                            ):
        """Initialize model."""
        self.node.get_network_state().update_round(source, round)






    ########################
    # CANDIDATES CALLBACKS #
    ########################
    def get_partial_gossipping_candidates(self):
        """Get the candidates from the train set to gossip the partial model."""
        def candidate_condition(node: str) -> bool:
            local_state = self.node.get_local_state()
            network_state = self.node.get_network_state()
            return set(local_state.train_set) - set(network_state.get_aggregation_sources(node))

        candidates = set(self.node.get_local_state().train_set) - {self.node.address}
        self.candidates = [n for n in candidates if len(candidate_condition(n)) != 0]
        logger.debug(self.node.address, f"📡 Candidates to gossip to: {self.candidates}")

    def get_full_gossipping_candidates(self):
        """Get the candidates from the train set to gossip the full model."""
        fixed_round = self.node.get_local_state().round
        def candidate_condition(node: str) -> bool:
            return self.node.get_network_state().get_round(node) < fixed_round

        self.candidates = [n for n in self.node.communication_protocol.get_neighbors(only_direct=True) if candidate_condition(n)]
        logger.debug(self.node.local_state.addr, f"📡 Candidates to gossip to: {self.candidates}")

    ######################
    # LOGGING CALLBACKS #
    ######################
    def finalize_logging(self, *args, **kwargs):
        """Logging callback."""
        logger.debug(self.node.address, f"🏃 Running stage: {(self.state)}")
        #self.event_log.append(self.event)
        self.state_log.append(self.state)

    def test(self, *args, **kwargs):
        """Test function for debugging."""
        logger.info(self.node.address, "Test function called.")
