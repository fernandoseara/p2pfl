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

import asyncio
from typing import TYPE_CHECKING

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
from p2pfl.stages.workflows.workflows import TrainingWorkflow

if TYPE_CHECKING:  # Only imports the below statements during type checking
    from p2pfl.node import Node


class BasicDFLWorkflow(TrainingWorkflow):
    """
    Class to run a federated learning workflow with transitions library.

    This class is used to define the states and transitions of the workflow.
    """

    def __init__(self, node: Node):
        """Initialize the workflow."""
        self.candidates: list[str] = []
        print(a)

        # Define states and events
        states = [
            {'name': "waiting_for_training_start"},
            {'name': 'training', 'parallel':
            [
                {'name': 'workflow', 'initial': 'starting_training', 'children':
                [
                    {'name': "starting_training", 'on_enter': 'on_enter_starting_training'},
                    {'name': "waiting_for_synchronization", 'on_enter': 'on_enter_waiting_for_synchronization'},
                    {'name': "nodes_synchronized", 'on_enter': 'on_enter_nodes_synchronized'},
                    {'name': "waiting_for_full_model"},
                    {'name': "updating_round", 'on_enter': 'on_enter_updating_round'},
                    {'name': "gossiping_full_model", 'on_enter': 'on_enter_gossipping_full_model'},
                    {'name': "waiting_for_network_start", 'on_timeout':"gossip_timeout"},
                    {'name': "round_initialized", 'on_enter': 'on_enter_round_initialized'},
                    {'name': 'p2p_voting', 'initial': 'starting_voting', 'on_final': 'on_final_p2p_voting', 'children':
                    [
                        {'name': "starting_voting", 'on_enter': 'on_enter_starting_voting'},
                        {'name': "voting", 'on_enter': 'on_enter_voting'},
                        {'name': "waiting_voting", 'timeout':Settings.training.VOTE_TIMEOUT, 'on_timeout':"voting_timeout"},
                        {'name': 'voting_finished', 'on_enter': 'on_enter_voting_finished', 'final':True},
                    ]},
                    {'name': 'p2p_learning', 'initial': 'evaluating', 'children':
                    [
                        {'name': 'evaluating', 'on_enter': 'on_enter_evaluating'},
                        {'name': 'training', 'on_enter': 'on_enter_training'},
                        {'name': 'gossipping_partial_aggregation', 'on_enter': 'on_enter_gossipping_partial_aggregation'},
                        {'name': "waiting_for_partial_aggregation", 'timeout':Settings.training.AGGREGATION_TIMEOUT, 'on_timeout':"aggregation_timeout"},
                        {'name': 'aggregating', 'on_enter': 'on_enter_aggregating'},
                        {'name': "aggregation_finished", 'on_enter': 'on_enter_aggregation_finished', 'final':True}
                    ]},
                    {'name': "round_finished", 'on_enter': 'on_enter_round_finished'},
                ]},
                {'name': 'event_handler', 'initial': 'waiting_network_start', 'children':
                [
                    {'name': "waiting_network_start", 'on_enter': 'on_enter_waiting_network_start'},
                    {'name': 'waiting_model_update', 'parallel':
                    [
                        {'name': 'waiting_round', 'initial': 'round', 'children':
                        [
                            {'name': "round", 'on_enter': 'on_enter_waiting_round_update'},
                            {'name': "rounds_updated" , 'final':True},
                        ]},
                        {'name': 'waiting_full_model', 'initial': 'full_model', 'children':
                        [
                            {'name': "full_model", 'on_enter': 'on_enter_waiting_full_model'},
                            {'name': "full_models_updated" , 'final':True},
                        ]},
                    ], 'on_final': "send_models_ready"},
                    {'name': "waiting_vote", 'on_enter': 'on_enter_waiting_vote'},
                    {'name': "waiting_partial_model", 'on_enter': 'on_enter_waiting_partial_model'},
                ]},
            ]},
            {'name': "training_finished", 'on_enter': 'on_enter_training_finished' , 'final':True}
        ]

        transitions = [
            # Event handlers
            {'trigger': 'node_started', 'source': 'training↦event_handler↦waiting_network_start', \
                'dest': 'training↦event_handler↦waiting_model_update', 'prepare': 'create_peer', \
                'conditions': 'is_all_nodes_started', 'after': 'send_network_ready'},

            {'trigger': 'peer_round_updated', 'source': 'round', \
                'dest': 'rounds_updated', 'prepare': 'save_peer_round_updated', \
                'conditions': 'is_all_models_initialized', 'after': 'send_peers_ready'},
            {'trigger': 'full_model_received', 'source': 'full_model', \
                'dest': 'full_models_updated', 'prepare': 'save_full_model', \
                'after': 'send_full_model_ready'},

            {'trigger': 'models_ready', 'source': 'training↦event_handler↦waiting_model_update', \
                'dest': 'training↦event_handler↦waiting_vote', \
                'conditions': 'in_train_set'},
            {'trigger': 'models_ready', 'source': 'training↦event_handler↦waiting_model_update', \
                'dest': None},

            {'trigger': 'vote', 'source': 'training↦event_handler↦waiting_vote', \
                'dest': 'training↦event_handler↦waiting_partial_model', 'prepare': 'save_votes', \
                'conditions': 'is_all_votes_received', 'after': 'send_votes_ready'},

            {'trigger': 'aggregate', 'source': 'training↦event_handler↦waiting_partial_model', \
                'dest': 'training↦event_handler↦waiting_model_update', 'prepare': 'save_aggregation', \
                'conditions': 'is_all_models_received', 'after': 'send_aggregation_ready'},

            # Setup & Initial synchronization
            {'trigger': 'start_learning', 'source': 'waiting_for_training_start', 'dest': 'training', 'after': 'set_model_initialized'},
            {'trigger': 'peer_learning_initiated', 'source': 'waiting_for_training_start', 'dest': 'training'},
            {'trigger': 'next_stage', 'source': 'training↦workflow↦starting_training', 'dest': 'training↦workflow↦waiting_for_synchronization'},
            {'trigger': 'network_ready', 'source': 'training↦workflow↦waiting_for_synchronization', 'dest': 'training↦workflow↦nodes_synchronized'},

            # Model initialization
            {'trigger': 'next_stage', 'source': 'training↦workflow↦nodes_synchronized', 'dest': 'training↦workflow↦updating_round', 'conditions': 'is_model_initialized'},
            {'trigger': 'next_stage', 'source': 'training↦workflow↦nodes_synchronized', 'dest': 'training↦workflow↦waiting_for_full_model'},
            {'trigger': 'full_model_ready', 'source': 'training↦workflow↦waiting_for_full_model', 'dest': 'training↦workflow↦updating_round'},

            # Update round
            {'trigger': 'continue_p2p_round_initialization', 'source': 'training↦workflow↦updating_round', 'dest': 'training↦workflow↦gossiping_full_model', 'prepare': ['get_full_gossipping_candidates'], 'conditions': 'candidate_exists'},
            {'trigger': 'continue_p2p_round_initialization', 'source': 'training↦workflow↦updating_round', 'dest': 'training↦workflow↦waiting_for_network_start'},

            # Gossip full model
            {'trigger': 'continue_p2p_round_initialization', 'source': 'training↦workflow↦gossiping_full_model', 'dest': 'training↦workflow↦round_initialized', 'conditions': 'is_all_models_initialized'},
            {'trigger': 'continue_p2p_round_initialization', 'source': 'training↦workflow↦gossiping_full_model', 'dest': 'training↦workflow↦waiting_for_network_start'},
            {'trigger': 'peers_ready', 'source': 'training↦workflow↦waiting_for_network_start', 'dest': 'training↦workflow↦round_initialized'},

            # Checking workflow finished
            {'trigger': 'next_stage', 'source': 'training↦workflow↦round_initialized', 'dest': 'training_finished', 'conditions': 'is_total_rounds_reached'},
            {'trigger': 'next_stage', 'source': 'training↦workflow↦round_initialized', 'dest': 'training↦workflow↦p2p_voting'},

            # Voting process
            {'trigger': 'continue_p2p_voting', 'source': 'training↦workflow↦p2p_voting↦starting_voting', 'dest': 'training↦workflow↦p2p_voting↦voting'},
            {'trigger': 'continue_p2p_voting', 'source': 'training↦workflow↦p2p_voting↦voting', 'dest': 'training↦workflow↦p2p_voting↦waiting_voting'},

            {'trigger': 'votes_ready', 'source': 'training↦workflow↦p2p_voting', 'dest': 'training↦workflow↦p2p_voting↦voting_finished'},
            {'trigger': 'voting_timeout', 'source': 'training↦workflow↦p2p_voting↦waiting_voting', 'dest': 'training↦workflow↦p2p_voting↦voting_finished'},

            # Training decision
            {'trigger': 'next_stage', 'source': 'training↦workflow↦p2p_voting', 'dest': 'training↦workflow↦p2p_learning',  'conditions': 'in_train_set'},
            {'trigger': 'next_stage', 'source': 'training↦workflow↦p2p_voting', 'dest': 'training↦workflow↦waiting_for_full_model'},

            # P2P learning flow
            {'trigger': 'continue_p2p_learning', 'source': 'training↦workflow↦p2p_learning↦evaluating', 'dest': 'training↦workflow↦p2p_learning↦training'},

            {'trigger': 'continue_p2p_learning', 'source': 'training↦workflow↦p2p_learning↦training', 'dest': 'training↦workflow↦p2p_learning↦gossipping_partial_aggregation', 'prepare': ['get_partial_gossipping_candidates'], 'conditions': 'candidate_exists'},
            {'trigger': 'continue_p2p_learning', 'source': 'training↦workflow↦p2p_learning↦training', 'dest': 'training↦workflow↦p2p_learning↦waiting_for_partial_aggregation'},
            {'trigger': 'continue_p2p_learning', 'source': 'training↦workflow↦p2p_learning↦gossipping_partial_aggregation', 'dest': 'training↦workflow↦p2p_learning↦waiting_for_partial_aggregation'},

            {'trigger': 'aggregation_ready', 'source': 'training↦workflow↦p2p_learning', 'dest': 'training↦workflow↦p2p_learning↦aggregating'},
            {'trigger': 'aggregation_timeout', 'source': 'training↦workflow↦p2p_learning↦waiting_for_partial_aggregation', 'dest': 'training↦workflow↦p2p_learning↦aggregating'},

            {'trigger': 'continue_p2p_learning', 'source': 'training↦workflow↦p2p_learning↦aggregating', 'dest': 'training↦workflow↦p2p_learning↦aggregation_finished'},

            # Loop
            {'trigger': 'next_stage', 'source': 'training↦workflow↦p2p_learning', 'dest': 'training↦workflow↦round_finished'},
            {'trigger': 'next_stage', 'source': 'training↦workflow↦round_finished', 'dest': 'training↦workflow↦waiting_for_full_model'},

        ]

        super().__init__(
            node=node,
            model=self,
            states=states,
            transitions=transitions,
        )

    @property
    def finished(self) -> bool:
        """
        Check if the workflow is finished.

        Returns:
            bool: True if the workflow is finished, False otherwise.

        """
        return self.is_training_finished()

    


    


    #####################
    # NETWORK CALLBACKS #
    #####################

    async def set_model_initialized(self, *args, **kwargs):
        """Set the model initialized."""
        # Set the model initialized
        self.node.get_learner().get_model().set_round(0)

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

    async def broadcast_start_learning(self,
                                        experiment_name: str,
                                        rounds: int,
                                        epochs: int,
                                        trainset_size: int
                                        ):
        """Broadcast start learning (from local)."""
        await BroadcastStartLearningStage.execute(
            experiment_name=experiment_name,
            epochs=epochs,
            trainset_size=trainset_size,
            total_rounds=rounds,
            node=self.node,
        )

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


    #####################
    # PREPARE CALLBACKS #
    #####################
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

# if __name__ == "__main__":
#     m = BasicDFLWorkflow(None)

#     experiment_name = "test"
#     rounds = 10
#     epochs = 10
#     trainset_size = 4

#     asyncio.get_event_loop().run_until_complete(m.start_learning(experiment_name=experiment_name, rounds=rounds, epochs=epochs, trainset_size=trainset_size))

#     print(m.is_starting_training())

if __name__ == "__main__":
    m = BasicDFLWorkflow(None)

    m.get_graph().draw('my_state_diagram.png', prog='dot')
