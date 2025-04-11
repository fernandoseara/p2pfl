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

from typing import TYPE_CHECKING

from transitions.extensions.asyncio import AsyncMachine, AsyncTimeout
from transitions.extensions.states import add_state_features

from p2pfl.communication.commands.message.models_agregated_command import ModelsAggregatedCommand
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.stages.base_node.aggregating_vote_train_set_stage import AggregatingVoteTrainSetStage
from p2pfl.stages.base_node.aggregation_finished_stage import AggregationFinishedStage
from p2pfl.stages.base_node.broadcast_start_learning_stage import BroadcastStartLearningStage
from p2pfl.stages.base_node.evaluate_stage import EvaluateStage
from p2pfl.stages.base_node.gossip_initial_model import GossipInitialModelStage
from p2pfl.stages.base_node.gossip_final_model_stage import GossipFinalModelStage
from p2pfl.stages.base_node.initialize_model_stage import InitializeModelStage
from p2pfl.stages.base_node.round_finished_stage import RoundFinishedStage
from p2pfl.stages.base_node.start_learning_stage import StartLearningStage
from p2pfl.stages.base_node.train_stage import TrainStage
from p2pfl.stages.base_node.gossip_partial_model_stage import GossipPartialModelStage
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

        self.candidates: list[str] = []

        # Define states and events
        states = [{'name': "waiting_for_training_start"},
            {'name': "starting_training"},
            {'name': "gossiping_initial_model"},
            {'name': "initializing_model"},
            {'name': "voting"},
            {'name': "waiting_voting", 'timeout':Settings.training.VOTE_TIMEOUT, 'on_timeout':"voting_timeout"},
            {'name': "evaluating"},
            {'name': "training"},
            {'name': "gossiping_partial_model"},
            {'name': "aggregating"},
            {'name': "waiting_for_aggregation",'timeout':Settings.training.AGGREGATION_TIMEOUT, 'on_timeout':"aggregation_timeout"},
            {'name': "gossiping_full_model"},
            {'name': "round_finished"},
            {'name': "training_finished", 'final':True}
        ]

        transitions = [
            {'trigger': 'send_starting_learning', 'source': 'waiting_for_training_start', 'dest': None, 'after': 'broadcast_start_learning'},
            {'trigger': 'start_learning', 'source': 'waiting_for_training_start', 'dest': 'starting_training'},
            {'trigger': 'gossip', 'source': 'starting_training', 'dest': 'gossiping_initial_model', 'prepare': ['get_initial_candidates'], 'conditions': 'candidate_exists'},
            
            {'trigger': 'finish_voting', 'source': 'gossiping_initial_model', 'dest': 'initializing_model'},
            {'trigger': 'vote', 'source': 'waiting_voting', 'dest': 'aggregating_voting', 'conditions': 'is_all_votes_received'},
            {'trigger': 'vote', 'source': 'waiting_voting', 'dest': None, 'after': 'save_votes'},
            {'trigger': 'voting_timeout', 'source': 'waiting_voting', 'dest': 'aggregating_voting'},
            {'trigger': 'train', 'source': 'aggregating_voting', 'dest': 'evaluating',  'conditions': 'in_train_set'},
            {'trigger': 'train', 'source': 'aggregating_voting', 'dest': 'aggregating', 'conditions': '!in_train_set'},
            {'trigger': 'train', 'source': 'evaluating', 'dest': 'training'},
            
            {'trigger': 'gossip', 'source': 'gossiping_partial_model', 'dest': None,  'prepare': ['get_partial_gossiping_candidates'], 'conditions': 'candidate_exists'},
            {'trigger': 'gossip', 'source': 'training', 'dest': 'gossiping_partial_model'},

            {'trigger': 'aggregate', 'source': 'waiting_for_aggregation', 'dest': None, 'conditions': 'is_missing_models', 'after': 'save_aggregation'},
            {'trigger': 'aggregate', 'source': 'waiting_for_aggregation', 'dest': 'gossiping_full_model'},
            {'trigger': 'aggregation_timeout', 'source': 'waiting_for_aggregation', 'dest': 'gossiping_full_model'},
            
            {'trigger': 'gossip', 'source': 'gossiping_full_model', 'dest': None,  'prepare': ['get_full_gossiping_candidates'], 'conditions': 'candidate_exists'},
            {'trigger': 'finish_round', 'source': 'gossiping_full_model', 'dest': 'round_finished'},
            {'trigger': 'step', 'source': 'round_finished', 'dest': 'voting', 'conditions': '!is_total_rounds_reached'},
            {'trigger': 'step', 'source': 'round_finished', 'dest': 'training_finished', 'conditions': 'is_total_rounds_reached'}
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
                                         rounds: int,
                                         epochs: int,
                                         trainset_size: int):
        """Start the training."""
        self.is_running = True
        await StartLearningStage.execute(
            experiment_name=experiment_name,
            rounds=rounds,
            epochs=epochs,
            trainset_size=trainset_size,
            node=self.node,
        )
        await self.gossip()


    async def on_enter_initial_gossiping(self):
        """Start the training."""
        await GossipInitialModelStage.execute(
            candidates=self.candidates,
            node=self.node,
        )

    async def on_enter_voting(self):
        """Vote for the train set."""
        await VoteTrainSetStage.execute(
            node=self.node,
        )
        await self.waiting_voting()

    async def on_exit_waiting_voting(self):
        """Aggregate the votes."""
        await AggregatingVoteTrainSetStage.execute(
            node=self.node,
        )

    async def on_enter_evaluating(self):
        """Evaluate the model."""
        await EvaluateStage.execute(
            node=self.node,
        )
        await self.train()

    async def on_enter_training(self):
        """Train the model."""
        await TrainStage.execute(
            node=self.node,
        )
        await self.gossip()

    async def on_enter_gossipping_partial_model(self):
        """Train the model."""
        await GossipPartialModelStage.execute(
            candidates=self.candidates,
            node=self.node,
        )
        await self.aggregate()

    async def on_exit_waiting_for_aggregation(self):
        """Finish the aggregation."""
        await AggregationFinishedStage.execute(
            node=self.node,
        )

    async def on_enter_gossiping_full_model(self):
        """Gossip the model."""
        await GossipFinalModelStage.execute(
            node=self.node,
        )
        await self.finish_round()

    async def on_enter_round_finished(self):
        """Finish the round."""
        await RoundFinishedStage.execute(
            node=self.node,
        )
        await self.step()

    async def on_enter_training_finished(self):
        """Finish the training."""
        await TrainingFinishedStage.execute(
            node=self.node,
        )
        self.is_running = False


    #######################
    # TRANSITION CALLBACK #
    #######################

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

    async def save_votes(self, source: str, tmp_votes: dict[str, int]):
        """Save the votes."""
        self.node.get_network_state().add_vote(source, tmp_votes)

    async def save_aggregation(self, model: P2PFLModel):
        """Save the aggregation."""
        models_added = self.node.get_aggregator().add_model(model)
        if models_added != []:
            # Communicate Aggregation
            self.node.get_communication_protocol().broadcast(
                self.node.get_communication_protocol().build_msg(
                    ModelsAggregatedCommand.get_name(),
                    models_added,
                    round=self.node.get_local_state().round,
                )
            )

    #####################
    # PREPARE CALLBACKS #
    #####################
    def get_initial_candidates(self):
        """Get the candidates for the initial gossiping."""
        def candidate_condition(node: str) -> bool:
            return node not in self.node.local_state.nei_status

        self.candidates = [n for n in self.node.communication_protocol.get_neighbors(only_direct=True) if candidate_condition(n)]
        logger.debug(self.node.local_state.addr, f"📡 Candidates to gossip to: {self.candidates}")

    def get_partial_gossiping_candidates(self):
        """Get the candidates from the train set to gossip the partial model."""
        def candidate_condition(node: str) -> bool:
            return set(self.node.local_state.train_set) - set(self.node.local_state.models[node].get_contributors())
        candidates = set(self.node.local_state.train_set) - {self.node.local_state.addr}
        self.candidates = [n for n in candidates if len(candidate_condition(n)) != 0]
        logger.debug(self.node.local_state.addr, f"📡 Candidates to gossip to: {self.candidates}")

    def get_full_gossiping_candidates(self):
        """Get the candidates from the train set to gossip the full model."""
        fixed_round = self.node.local_state.round
        def candidate_condition(node: str) -> bool:
            return self.node.local_state.nei_status[node] < fixed_round

        self.candidates = [n for n in self.node.communication_protocol.get_neighbors(only_direct=True) if candidate_condition(n)]
        logger.debug(self.node.local_state.addr, f"📡 Candidates to gossip to: {self.candidates}")

    ##############
    # CONDITIONS #
    ##############
    def candidate_exists(self):
        """Check if there are candidates."""
        return len(self.candidates) > 0

    def is_all_votes_received(self):
        """Check if all votes from neis have been received."""
        nc_votes = {
                k: v
                for k, v in self.node.get_network_state().get_all_votes().items()
                if k in list(self.node.communication_protocol.get_neighbors(only_direct=False)) or k == self.node.address
            }

        needed_votes = set(list(self.node.communication_protocol.get_neighbors(only_direct=False)) + [self.node.address])
        return needed_votes == set(nc_votes.keys())

    def in_train_set(self):
        """Check if the node is in the train set."""
        return self.node.address in self.node.get_local_state().train_set

    def is_total_rounds_reached(self):
        """Check if the total rounds have been reached."""
        return self.node.get_local_state().round >= self.node.get_local_state().total_rounds

    def is_missing_models(self):
        """Check if there are missing models."""
        return len(self.node.get_local_state().train_set) > len(self.node.aggregator.get_aggregated_models())

if __name__ == "__main__":
    m = BasicDFLWorkflow(None)

    m.get_graph().draw('my_state_diagram.png', prog='dot')
