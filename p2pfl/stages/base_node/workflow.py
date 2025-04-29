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

from p2pfl.communication.commands.message.model_initialized_command import ModelInitializedCommand
from p2pfl.communication.commands.message.node_initialized_command import NodeInitializedCommand
from p2pfl.communication.commands.message.start_learning_command import StartLearningCommand
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.stages.base_node.aggregating_vote_train_set_stage import AggregatingVoteTrainSetStage
from p2pfl.stages.base_node.aggregation_finished_stage import AggregationFinishedStage
from p2pfl.stages.base_node.broadcast_start_learning_stage import BroadcastStartLearningStage
from p2pfl.stages.base_node.evaluate_stage import EvaluateStage
from p2pfl.stages.base_node.gossip_final_model_stage import GossipFinalModelStage
from p2pfl.stages.base_node.gossip_initial_model import GossipInitialModelStage
from p2pfl.stages.base_node.gossip_partial_model_stage import GossipPartialModelStage
from p2pfl.stages.base_node.initialize_model_stage import InitializeModelStage
from p2pfl.stages.base_node.round_finished_stage import RoundFinishedStage
from p2pfl.stages.base_node.start_learning_stage import StartLearningStage
from p2pfl.stages.base_node.train_stage import TrainStage
from p2pfl.stages.base_node.training_finished_stage import TrainingFinishedStage
from p2pfl.stages.base_node.vote_train_set_stage import VoteTrainSetStage
from p2pfl.stages.workflows import TrainingWorkflow

if TYPE_CHECKING:  # Only imports the below statements during type checking
    from p2pfl.node import Node


class BasicDFLWorkflow(TrainingWorkflow):
    """
    Class to run a federated learning workflow with transitions library.
    """

    def __init__(self, node: Node):
        """Initialize the workflow."""

        self.initial_node = False
        self.candidates: list[str] = []

        # Define states and events
        states = [
            {'name': "waiting_for_training_start"},
            {'name': "starting_training", 'on_enter': 'on_enter_starting_training'},
            {'name': "waiting_for_synchronization", 'on_enter': 'on_enter_waiting_for_synchronization'},
            {'name': "nodes_synchronized", 'on_enter': 'on_enter_nodes_synchronized'},
            {'name': "gossipping_initial_model", 'on_enter': 'on_enter_gossipping_initial_model'},
            {'name': "waiting_for_initial_model"},
            {'name': "initial_model_received", 'on_enter': 'on_enter_initial_model_received'},
            {'name': "waiting_for_network_start"},
            {'name': 'p2p_voting', 'initial': 'starting_voting', 'on_final': 'on_final_p2p_voting', 'children':
             [
                {'name': "starting_voting", 'on_enter': 'on_enter_starting_voting'},
                {'name': "waiting_voting", 'timeout':Settings.training.VOTE_TIMEOUT, 'on_timeout':"voting_timeout", 'on_enter': 'on_enter_waiting_voting'},
                {'name': 'voting_finished', 'on_enter': 'on_enter_voting_finished', 'final':True},
            ]},
            {'name': 'p2p_learning', 'initial': 'evaluating', 'children':
             [
                {'name': 'evaluating', 'on_enter': 'on_enter_evaluating'},
                {'name': 'training', 'on_enter': 'on_enter_training'},
                {'name': 'gossipping_partial_aggregation', 'on_enter': 'on_enter_gossipping_partial_aggregation', 'on_exit': 'on_exit_waiting_for_partial_aggregation'},
                {'name': "waiting_for_partial_aggregation", 'timeout':Settings.training.AGGREGATION_TIMEOUT, 'on_timeout':"aggregation_timeout"},
                'aggregating', {'name': "aggregation_finished", 'on_enter': 'on_enter_aggregation_finished', 'final':True}
            ]},
            {'name': 'waiting_for_full_aggregation'},
            {'name': "gossiping_full_model", 'on_enter': 'on_enter_gossipping_full_model'},
            {'name': "round_finished", 'on_enter': 'on_enter_round_finished'},
            {'name': "training_finished", 'on_enter': 'on_enter_training_finished' , 'final':True}
        ]

        transitions = [
            # Setup
            {'trigger': 'start_learning', 'source': 'waiting_for_training_start', 'dest': 'starting_training', 'after': 'set_model_initialized'},
            {'trigger': 'peer_learning_initiated', 'source': 'waiting_for_training_start', 'dest': 'starting_training'},
            {'trigger': 'next_stage', 'source': 'starting_training', 'dest': 'waiting_for_synchronization'},

            # Initial synchronization
            {'trigger': 'node_started', 'source': 'waiting_for_synchronization', 'dest': 'nodes_synchronized', 'prepare': 'create_peer', 'conditions': 'is_all_nodes_started'},

            # Initial gossip
            {'trigger': 'next_stage', 'source': 'nodes_synchronized', 'dest': 'gossipping_initial_model', 'conditions': 'is_model_initialized'},
            {'trigger': 'next_stage', 'source': 'nodes_synchronized', 'dest': 'waiting_for_initial_model'},
            {'trigger': 'initial_model_received', 'source': 'waiting_for_initial_model', 'dest': 'initial_model_received', 'prepare': 'initialize_model', 'conditions': 'is_model_initialized'},
            {'trigger': 'next_stage', 'source': 'initial_model_received', 'dest': 'gossipping_initial_model'},

            # Initial gossip
            {'trigger': 'next_stage', 'source': 'gossipping_initial_model', 'dest': 'p2p_voting↦starting_voting', 'conditions': 'is_all_nodes_started'},
            {'trigger': 'next_stage', 'source': 'gossipping_initial_model', 'dest': 'waiting_for_network_start'},

            {'trigger': 'model_initialized', 'source': ['gossipping_initial_model','waiting_for_network_start'], 'dest': 'p2p_voting↦starting_voting', 'prepare': 'save_peer_model_initialized', 'conditions': 'is_all_models_initialized'},

            # Voting process
            {'trigger': 'continue_p2p_voting', 'source': 'p2p_voting↦starting_voting', 'dest': 'p2p_voting↦waiting_voting'},

            {'trigger': 'vote', 'source': 'p2p_voting', 'dest': 'p2p_voting↦voting_finished', 'prepare': 'save_votes', 'conditions': 'is_all_votes_received'},
            {'trigger': 'voting_timeout', 'source': 'p2p_voting↦waiting_voting', 'dest': 'p2p_voting↦voting_finished'},

            # Training decision
            {'trigger': 'next_stage', 'source': 'p2p_voting↦voting_finished', 'dest': 'p2p_learning',  'conditions': 'in_train_set'},
            {'trigger': 'next_stage', 'source': 'p2p_voting↦voting_finished', 'dest': 'waiting_for_full_aggregation'},

            # P2P learning flow
            {'trigger': 'continue_p2p_learning', 'source': 'p2p_learning↦evaluating', 'dest': 'p2p_learning↦training'},

            {'trigger': 'continue_p2p_learning', 'source': 'p2p_learning↦training', 'dest': 'p2p_learning↦gossipping_partial_aggregation', 'prepare': ['get_partial_gossipping_candidates'], 'conditions': 'candidate_exists'},
            {'trigger': 'continue_p2p_learning', 'source': 'p2p_learning↦training', 'dest': 'p2p_learning↦waiting_for_partial_aggregation'},
            {'trigger': 'continue_p2p_learning', 'source': 'p2p_learning↦gossipping_partial_aggregation', 'dest': 'p2p_learning↦waiting_for_partial_aggregation'},

            {'trigger': 'aggregate', 'source': 'p2p_learning↦waiting_for_partial_aggregation', 'dest': 'p2p_learning↦aggregating', 'conditions': 'is_all_models_received', 'prepare': 'save_aggregation'},
            {'trigger': 'aggregation_timeout', 'source': 'p2p_learning↦waiting_for_partial_aggregation', 'dest': 'p2p_learning↦aggregating'},

            {'trigger': 'next_stage', 'source': 'p2p_learning↦aggregating', 'dest': 'aggregation_finished'},

            # Receiving model externally
            {'trigger': 'full_aggregated_model_received', 'source': 'p2p_learning', 'dest': 'aggregation_finished'},
            {'trigger': 'full_aggregated_model_received', 'source': 'waiting_for_full_aggregation', 'dest': 'gossiping_full_model', 'prepare': ['get_full_gossipping_candidates'], 'conditions': 'candidate_exists'},

            # Gossip full model
            {'trigger': 'next_stage', 'source': 'aggregation_finished', 'dest': 'gossiping_full_model', 'prepare': ['get_full_gossipping_candidates'], 'conditions': 'candidate_exists'},
            {'trigger': 'next_stage', 'source': 'gossiping_full_model', 'dest': None,  'prepare': ['get_full_gossipping_candidates'], 'conditions': 'candidate_exists'},
            {'trigger': 'next_stage', 'source': 'gossiping_full_model', 'dest': 'round_finished'},
            {'trigger': 'next_stage', 'source': 'round_finished', 'dest': 'training_finished', 'conditions': 'is_total_rounds_reached'},

            # Next round / finish
            {'trigger': 'next_stage', 'source': 'round_finished', 'dest': 'waiting_voting'},

        ]

        super().__init__(
            node=node,
            model=self,
            states=states,
            transitions=transitions,
        )

    @property
    def finished(self) -> bool:
        """Return the name of the workflow."""
        #return self.training_finished.is_active
        return False

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
        logger.info(self.node.address, "Starting learning workflow.")
        self.is_running = True
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
        except Exception as e:
            logger.debug(self.node.address, f"Error broadcasting start learning command: {e}")

    async def on_enter_nodes_synchronized(self, source: str):
        """All nodes are synchronized."""
        logger.debug(self.node.address, "All nodes are synchronized.")
        await self.next_stage()

    async def on_enter_initial_model_received(self, source: str, weights: bytes):
        """Initialize the model."""
        await self.next_stage()

    async def on_enter_gossipping_initial_model(self):
        """Gossip the partial model."""
        self.get_initial_gossipping_candidates()

        await GossipInitialModelStage.execute(
            candidates=self.candidates,
            node=self.node,
        )

        await self.next_stage()

    async def on_enter_starting_voting(self, *args, **kwargs):
        """Set the model initialized."""
        await self.continue_p2p_voting()

    async def on_enter_waiting_voting(self):
        """Vote for the train set."""
        await VoteTrainSetStage.execute(
            node=self.node,
        )

        logger.debug(self.node.address, "⏳ Waiting other node votes.")

    async def on_enter_voting_finished(self, *args, **kwargs):
        """Finish the voting."""
        await AggregatingVoteTrainSetStage.execute(
            node=self.node,
        )

    async def on_final_p2p_voting(self, *args, **kwargs):
        """Finish the voting."""
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

    async def on_exit_p2p_learning(self):
        """Finish the training."""
        pass

    async def on_enter_aggregating(self):
        """Aggregate the models."""
        # Set aggregated model
        agg_model = self.node.get_aggregator().aggregate(self.node.get_network_state().get_all_models())
        self.node.get_learner().set_model(agg_model)

    async def on_exit_waiting_for_partial_aggregation(self):
        """Finish the aggregation."""
        await AggregationFinishedStage.execute(
            node=self.node,
        )
        await self.next_stage()

    async def on_enter_aggregation_finished(self):
        """Finish the aggregation."""
        await self.next_stage()

    async def on_enter_gossipping_full_model(self):
        """Gossip the model."""
        await GossipFinalModelStage.execute(
            node=self.node,
        )
        await self.next_stage()

    async def on_enter_round_finished(self):
        """Finish the round."""
        await RoundFinishedStage.execute(
            node=self.node,
        )
        await self.next_stage()

    async def on_enter_training_finished(self):
        """Finish the training."""
        await TrainingFinishedStage.execute(
            node=self.node,
        )
        self.is_running = False


    #####################
    # NETWORK CALLBACKS #
    #####################

    async def set_model_initialized(self, *args, **kwargs):
        """Set the initial node."""
        self.is_model_initialized = True

    async def initialize_model(self,
                            source: str,
                            weights: bytes):
        """Initialize model."""
        # Set source model round
        self.node.get_network_state().update_round(source, 0)

        await InitializeModelStage.execute(
            source=source,
            weights=weights,
            node=self.node,
        )
        self.is_model_initialized = True

    async def save_peer_model_initialized(self,
                            source: str):
        """Initialize model."""
        self.node.get_network_state().update_round(source, 0)

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

    async def save_aggregation(self, model: P2PFLModel):
        """Save the aggregation."""
        self.node.get_network_state().add_model(model)

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
    def get_initial_gossipping_candidates(self):
        """Get the candidates for the initial gossiping."""
        def candidate_condition(node: str) -> bool:
            round = self.node.get_network_state().get_round(node)
            return round < 0 if round else True

        self.candidates = [n for n in self.node.communication_protocol.get_neighbors(only_direct=True) if candidate_condition(n)]
        logger.debug(self.node.local_state.addr, f"📡 Candidates to gossip to: {self.candidates}")

    def get_partial_gossipping_candidates(self):
        """Get the candidates from the train set to gossip the partial model."""
        def candidate_condition(node: str) -> bool:
            local_state = self.node.get_local_state()
            network_state = self.node.get_network_state()
            return set(local_state.train_set) - set(network_state.get_aggregation_sources(node))
        candidates = set(self.node.get_local_state().train_set) - set(self.node.address)
        self.candidates = [n for n in candidates if len(candidate_condition(n)) != 0]
        logger.debug(self.node.address, f"📡 Candidates to gossip to: {self.candidates}")

    def get_full_gossipping_candidates(self):
        """Get the candidates from the train set to gossip the full model."""
        fixed_round = self.node.local_state.round
        def candidate_condition(node: str) -> bool:
            return self.node.local_state.nei_status[node] < fixed_round

        self.candidates = [n for n in self.node.communication_protocol.get_neighbors(only_direct=True) if candidate_condition(n)]
        logger.debug(self.node.local_state.addr, f"📡 Candidates to gossip to: {self.candidates}")

    ##############
    # CONDITIONS #
    ##############
    def is_model_initialized(self, *args, **kwargs):
        """Check if the model has been initialized."""
        return self.is_model_initialized

    def is_all_nodes_started(self, *args, **kwargs):
        """Check if all nodes have started."""
        return len(self.node.get_network_state().list_peers()) == (len(self.node.communication_protocol.get_neighbors(only_direct=True)) + 1)

    def is_all_models_initialized(self, *args, **kwargs):
        """Check if all models have been initialized."""
        rounds = self.node.get_network_state().get_all_rounds()
        return sum(1 for value in rounds.values() if value == 0) == (len(self.node.communication_protocol.get_neighbors(only_direct=True)) + 1)

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
        return len(self.node.get_local_state().train_set) == len(self.node.aggregator.get_aggregated_models())

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
