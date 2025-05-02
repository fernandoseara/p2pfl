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
"""Event handler workflow for the base node."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from transitions.extensions import HierarchicalAsyncGraphMachine
from transitions.extensions.asyncio import AsyncTimeout
from transitions.extensions.nesting import NestedState
from transitions.extensions.states import add_state_features

from p2pfl.management.logger import logger
from p2pfl.settings import Settings

from p2pfl.communication.commands.message.node_initialized_command import NodeInitializedCommand
from p2pfl.communication.commands.message.peer_round_updated_command import PeerRoundUpdatedCommand
from p2pfl.communication.commands.message.start_learning_command import StartLearningCommand
from p2pfl.learning.frameworks.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.stages.base_node.aggregating_vote_train_set_stage import AggregatingVoteTrainSetStage
from p2pfl.stages.base_node.broadcast_start_learning_stage import BroadcastStartLearningStage
from p2pfl.stages.base_node.evaluate_stage import EvaluateStage
from p2pfl.stages.base_node.gossip_full_model_stage import GossipFullModelStage
from p2pfl.stages.base_node.gossip_initial_model import GossipInitialModelStage
from p2pfl.stages.base_node.gossip_partial_model_stage import GossipPartialModelStage
from p2pfl.stages.base_node.start_learning_stage import StartLearningStage
from p2pfl.stages.base_node.train_stage import TrainStage
from p2pfl.stages.base_node.training_finished_stage import TrainingFinishedStage
from p2pfl.stages.base_node.update_round_stage import UpdateRoundStage
from p2pfl.stages.base_node.vote_train_set_stage import VoteTrainSetStage

if TYPE_CHECKING:
    from p2pfl.node import Node

NestedState.separator = '↦'

@add_state_features(AsyncTimeout)
class TimeoutMachine(HierarchicalAsyncGraphMachine):
    """State machine with timeout support."""

    pass

from p2pfl.management.logger import logger

states = [
    {'name': "starting_training", 'on_enter': 'on_enter_starting_training'},
    {'name': "waiting_for_synchronization", 'on_enter': 'on_enter_waiting_for_synchronization'},
    {'name': "nodes_synchronized", 'on_enter': 'on_enter_nodes_synchronized'},
    {'name': "waiting_for_full_model"},
    {'name': "updating_round", 'on_enter': 'on_enter_updating_round'},
    {'name': "gossiping_full_model", 'on_enter': 'on_enter_gossipping_full_model'},
    {'name': "waiting_for_network_start", 'on_timeout': "gossip_timeout"},
    {'name': "round_initialized", 'on_enter': 'on_enter_round_initialized'},
    {'name': 'p2p_voting', 'initial': 'starting_voting', 'on_final': 'on_final_p2p_voting', 'children': [
        {'name': "starting_voting", 'on_enter': 'on_enter_starting_voting'},
        {'name': "voting", 'on_enter': 'on_enter_voting'},
        {'name': "waiting_voting", 'timeout': Settings.training.VOTE_TIMEOUT, 'on_timeout': "voting_timeout"},
        {'name': 'voting_finished', 'on_enter': 'on_enter_voting_finished', 'final': True},
    ]},
    {'name': 'p2p_learning', 'initial': 'evaluating', 'children': [
        {'name': 'evaluating', 'on_enter': 'on_enter_evaluating'},
        {'name': 'training', 'on_enter': 'on_enter_training'},
        {'name': 'gossipping_partial_aggregation', 'on_enter': 'on_enter_gossipping_partial_aggregation'},
        {'name': "waiting_for_partial_aggregation", 'timeout': Settings.training.AGGREGATION_TIMEOUT, 'on_timeout': "aggregation_timeout"},
        {'name': 'aggregating', 'on_enter': 'on_enter_aggregating'},
        {'name': "aggregation_finished", 'on_enter': 'on_enter_aggregation_finished', 'final': True},
    ]},
    {'name': "round_finished", 'on_enter': 'on_enter_round_finished'},
]

transitions = [
    # Setup & Initial synchronization
    {'trigger': 'next_stage', 'source': 'starting_training', 'dest': 'waiting_for_synchronization'},
    {'trigger': 'network_ready', 'source': 'waiting_for_synchronization', 'dest': 'nodes_synchronized'},

    # Model initialization
    {'trigger': 'next_stage', 'source': 'nodes_synchronized', 'dest': 'updating_round', 'conditions': 'is_model_initialized'},
    {'trigger': 'next_stage', 'source': 'nodes_synchronized', 'dest': 'waiting_for_full_model'},
    {'trigger': 'full_model_ready', 'source': 'waiting_for_full_model', 'dest': 'updating_round'},

    # Update round
    {'trigger': 'continue_p2p_round_initialization', 'source': 'updating_round', 'dest': 'gossiping_full_model', 
     'prepare': ['get_full_gossipping_candidates'], 'conditions': 'candidate_exists'},
    {'trigger': 'continue_p2p_round_initialization', 'source': 'updating_round', 'dest': 'waiting_for_network_start'},

    # Gossip full model
    {'trigger': 'continue_p2p_round_initialization', 'source': 'gossiping_full_model', 'dest': 'round_initialized', 
     'conditions': 'is_all_models_initialized'},
    {'trigger': 'continue_p2p_round_initialization', 'source': 'gossiping_full_model', 'dest': 'waiting_for_network_start'},
    {'trigger': 'peers_ready', 'source': 'waiting_for_network_start', 'dest': 'round_initialized'},

    # Workflow finish check
    {'trigger': 'next_stage', 'source': 'round_initialized', 'dest': 'training_finished', 'conditions': 'is_total_rounds_reached'},
    {'trigger': 'next_stage', 'source': 'round_initialized', 'dest': 'p2p_voting'},

    # Voting
    {'trigger': 'continue_p2p_voting', 'source': 'p2p_voting↦starting_voting', 'dest': 'p2p_voting↦voting'},
    {'trigger': 'continue_p2p_voting', 'source': 'p2p_voting↦voting', 'dest': 'p2p_voting↦waiting_voting'},
    {'trigger': 'votes_ready', 'source': 'p2p_voting', 'dest': 'p2p_voting↦voting_finished'},
    {'trigger': 'voting_timeout', 'source': 'p2p_voting↦waiting_voting', 'dest': 'p2p_voting↦voting_finished'},

    # Voting outcome
    {'trigger': 'next_stage', 'source': 'p2p_voting', 'dest': 'p2p_learning', 'conditions': 'in_train_set'},
    {'trigger': 'next_stage', 'source': 'p2p_voting', 'dest': 'waiting_for_full_model'},

    # Learning
    {'trigger': 'continue_p2p_learning', 'source': 'p2p_learning↦evaluating', 'dest': 'p2p_learning↦training'},
    {'trigger': 'continue_p2p_learning', 'source': 'p2p_learning↦training', 'dest': 'p2p_learning↦gossipping_partial_aggregation', 
     'prepare': ['get_partial_gossipping_candidates'], 'conditions': 'candidate_exists'},
    {'trigger': 'continue_p2p_learning', 'source': 'p2p_learning↦training', 'dest': 'p2p_learning↦waiting_for_partial_aggregation'},
    {'trigger': 'continue_p2p_learning', 'source': 'p2p_learning↦gossipping_partial_aggregation', 'dest': 'p2p_learning↦waiting_for_partial_aggregation'},

    {'trigger': 'aggregation_ready', 'source': 'p2p_learning', 'dest': 'p2p_learning↦aggregating'},
    {'trigger': 'aggregation_timeout', 'source': 'p2p_learning↦waiting_for_partial_aggregation', 'dest': 'p2p_learning↦aggregating'},

    {'trigger': 'continue_p2p_learning', 'source': 'p2p_learning↦aggregating', 'dest': 'p2p_learning↦aggregation_finished'},

    # Loop
    {'trigger': 'next_stage', 'source': 'p2p_learning', 'dest': 'round_finished'},
    {'trigger': 'next_stage', 'source': 'round_finished', 'dest': 'waiting_for_full_model'},
]


class TrainingModel(object):
    """
    Event handler model for the base node.

    This class is used to handle the events of the base node.
    It uses the transitions library to create a state machine.

    """
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

    async def on_exit_p2p_learning(self):
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

event_handler = TimeoutMachine(states=states, transitions=transitions, initial='waiting_network_start')
