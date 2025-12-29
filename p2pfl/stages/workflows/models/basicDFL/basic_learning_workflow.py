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

from p2pfl.communication.commands.message.node_initialized_command import NodeInitializedCommand
from p2pfl.communication.commands.message.peer_round_updated_command import PeerRoundUpdatedCommand
from p2pfl.communication.commands.message.start_learning_command import StartLearningCommand
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.stages.base_node import (
    AggregatingVoteTrainSetStage,
    EvaluateStage,
    GossipFullModelStage,
    GossipPartialModelStage,
    TrainStage,
    UpdateRoundStage,
    VoteTrainSetStage,
)
from p2pfl.stages.base_node.start_learning_stage import StartLearningStage
from p2pfl.stages.network_state.basic_network_state import BasicNetworkState
from p2pfl.stages.workflows.models.learning_workflow_model import LearningWorkflowModel
from p2pfl.utils.pytransitions import StateAdapter, TransitionAdapter

if TYPE_CHECKING:
    from p2pfl.node import Node

def get_states() -> list[dict]:
    """Define the states for the workflow."""
    states = [
        # Setup & initial synchronization
        StateAdapter(name="waitingSetup"),
        StateAdapter(name="startingTraining"),
        StateAdapter(name="waitingForSynchronization"),
        StateAdapter(name="nodesSynchronized"),
        StateAdapter(name="waitingForFullModel"),
        StateAdapter(name="updatingRound"),
        StateAdapter(name="gossipingFullModel"),
        StateAdapter(name="waitingForNetworkStart"),
        StateAdapter(name="roundInitialized"),
        StateAdapter(name='p2pVoting', initial='startingVoting', on_final='on_final_p2p_voting', children=[
            StateAdapter(name="startingVoting"),
            StateAdapter(name="voting"),
            StateAdapter(name="waitingVoting", timeout=Settings.training.VOTE_TIMEOUT, on_timeout="voting_timeout"),
            StateAdapter(name='votingFinished', final=True),
        ]),
        StateAdapter(name='p2pLearning', initial='evaluating', on_final='on_final_p2p_learning', children=[
            StateAdapter(name='evaluating'),
            StateAdapter(name='training'),
            StateAdapter(name='gossippingPartialAggregation'),
            StateAdapter(name="waitingForPartialAggregation", timeout=Settings.training.AGGREGATION_TIMEOUT, on_timeout="aggregation_timeout"),
            StateAdapter(name='aggregating'),
            StateAdapter(name="aggregationFinished", final=True),
        ]),
        StateAdapter(name="roundFinished"),
        StateAdapter(name="trainingFinished", final=True),
    ]

    return [state.to_dict() for state in states]

def get_transitions() -> list[dict]:
    """Define the transitions for the workflow."""
    transitions = [
        # Setup & initial synchronization
        TransitionAdapter(trigger='setup', source='waitingSetup', dest='startingTraining'),
        TransitionAdapter(trigger='next_stage', source='startingTraining', dest='waitingForSynchronization'),
        TransitionAdapter(trigger='network_ready', source='waitingForSynchronization', dest='nodesSynchronized'),

        # Model initialization
        TransitionAdapter(trigger='next_stage', source='nodesSynchronized', dest='updatingRound', conditions='is_model_initialized'),
        TransitionAdapter(trigger='next_stage', source='nodesSynchronized', dest='waitingForFullModel'),
        TransitionAdapter(trigger='full_model_ready', source='waitingForFullModel', dest='updatingRound'),

        # Update round
        TransitionAdapter(trigger='continue_p2p_round_initialization', source='updatingRound', dest='gossipingFullModel',
        prepare=['get_full_gossipping_candidates'], conditions='candidate_exists'),
        TransitionAdapter(trigger='continue_p2p_round_initialization', source='updatingRound', dest='waitingForNetworkStart'),

        # Gossip full model
        TransitionAdapter(trigger='continue_p2p_round_initialization', source='gossipingFullModel', dest='roundInitialized',
        conditions='is_all_models_initialized'),
        TransitionAdapter(trigger='continue_p2p_round_initialization', source='gossipingFullModel', dest='waitingForNetworkStart'),
        TransitionAdapter(trigger='peers_ready', source='waitingForNetworkStart', dest='roundInitialized'),

        # Workflow finish check
        TransitionAdapter(trigger='next_stage', source='roundInitialized', dest='trainingFinished', conditions='is_total_rounds_reached'),
        TransitionAdapter(trigger='next_stage', source='roundInitialized', dest='p2pVoting'),
        # Voting
        TransitionAdapter(trigger='continue_p2p_voting', source='p2pVoting_startingVoting', dest='p2pVoting_voting'),
        TransitionAdapter(trigger='continue_p2p_voting', source='p2pVoting_voting', dest='p2pVoting_waitingVoting'),
        TransitionAdapter(trigger='votes_ready', source='p2pVoting', dest='p2pVoting_votingFinished'),
        TransitionAdapter(trigger='voting_timeout', source='p2pVoting_waitingVoting', dest='p2pVoting_votingFinished'),

        # Voting outcome
        TransitionAdapter(trigger='next_stage', source='p2pVoting', dest='p2pLearning', conditions='in_train_set'),
        TransitionAdapter(trigger='next_stage', source='p2pVoting', dest='waitingForFullModel'),
        # Learning
        TransitionAdapter(trigger='continue_p2p_learning', source='p2pLearning_evaluating', dest='p2pLearning_training'),
        TransitionAdapter(trigger='continue_p2p_learning', source='p2pLearning_training', dest='p2pLearning_gossippingPartialAggregation',
        prepare=['get_partial_gossipping_candidates'], conditions='candidate_exists'),
        TransitionAdapter(trigger='continue_p2p_learning', source='p2pLearning_training', dest='p2pLearning_waitingForPartialAggregation'),
        TransitionAdapter(trigger='continue_p2p_learning', source='p2pLearning_gossippingPartialAggregation',
            dest='p2pLearning_waitingForPartialAggregation'),

        TransitionAdapter(trigger='aggregation_ready', source='p2pLearning', dest='p2pLearning_aggregating'),
        TransitionAdapter(trigger='aggregation_timeout', source='p2pLearning_waitingForPartialAggregation',
            dest='p2pLearning_aggregating'),
        TransitionAdapter(trigger='continue_p2p_learning', source='p2pLearning_aggregating', dest='p2pLearning_aggregationFinished'),
        # Loop
        TransitionAdapter(trigger='next_stage', source='p2pLearning', dest='roundFinished'),
        TransitionAdapter(trigger='next_stage', source='roundFinished', dest='updatingRound', conditions='is_all_models_received'),
        TransitionAdapter(trigger='next_stage', source='roundFinished', dest='waitingForFullModel'),
    ]

    return [transition.to_dict() for transition in transitions]

class BasicLearningWorkflowModel(LearningWorkflowModel):
    """Model for the training workflow."""

    def __init__(self, node: Node, network_state: BasicNetworkState) -> None:
        """Initialize the workflow model."""
        self.network_state: BasicNetworkState = network_state

        super().__init__(
            node=node,
        )

    ########################################
    # EVENTS (Overridden by pytransitions) #
    ########################################
    async def next_stage(self) -> bool:
        """Handle the next stage event."""
        raise RuntimeError("Should be overridden!")

    async def network_ready(self) -> bool:
        """Handle the network ready event."""
        raise RuntimeError("Should be overridden!")

    async def continue_p2p_round_initialization(self) -> bool:
        """Handle the continue p2p round initialization event."""
        raise RuntimeError("Should be overridden!")

    async def continue_p2p_voting(self) -> bool:
        """Handle the continue p2p voting event."""
        raise RuntimeError("Should be overridden!")

    async def continue_p2p_learning(self) -> bool:
        """Handle the continue p2p learning event."""
        raise RuntimeError("Should be overridden!")

    ###################
    # STATE CALLBACKS #
    ###################
    async def on_enter_startingTraining(
        self,
        experiment_name: str,
        rounds: int = 0,
        epochs: int = 0,
        trainset_size: int = 0,
        source: str | None = None
        ):
        """Start the training."""
        communication_protocol = self.node.get_communication_protocol()
        local_state = self.node.get_local_state()

        logger.info(self.node.address, "⏳ Starting training.")
        await StartLearningStage.execute(
            experiment_name=experiment_name,
            rounds=rounds,
            epochs=epochs,
            trainset_size=trainset_size,
            node=self.node,
        )

        # Comunicate start learning command
        await communication_protocol.broadcast_gossip(
            communication_protocol.build_msg(
                StartLearningCommand.get_name(),
                [
                    local_state.total_rounds,
                    self.node.get_learner().get_epochs(),
                    local_state.get_experiment().trainset_size,
                    local_state.get_experiment().exp_name,
                    self.node.get_node_workflow().get_workflow_type().value
                ]
            )
        )


        await self.next_stage()

    async def on_enter_waitingForSynchronization(self):
        """Wait for the synchronization."""
        communication_protocol = self.node.get_communication_protocol()

        # Wait and gossip model initialization
        logger.info(self.node.address, "⏳ Waiting initialization.")

        # Set self model initialized
        await self.node.get_event_handler().node_started(
            source=self.node.address,
        )

        # Communicate Initialization
        try:
            await communication_protocol.broadcast_gossip(communication_protocol.build_msg(NodeInitializedCommand.get_name()))

        except Exception as e:
            logger.debug(self.node.address, f"Error broadcasting start learning command: {e}")

    async def on_enter_nodesSynchronized(self, *args, **kwargs):
        """All nodes are synchronized."""
        logger.debug(self.node.address, "🤝 All nodes synchronized.")
        await self.next_stage()

    async def on_enter_roundInitialized(self, *args, **kwargs):
        """Round initialized."""
        logger.debug(self.node.address, "🤖 Round initialized.")
        await self.next_stage()

    async def on_enter_updatingRound(self, *args, **kwargs):
        """Update the round."""
        # Set self model round
        await UpdateRoundStage.execute(
            network_state=self.network_state,
            node=self.node,
        )

        # Send event to the workflow
        await self.node.get_event_handler().peer_round_updated(
            self.node.address, self.node.get_local_state().round
        )

        # Communicate round update
        await self.node.get_communication_protocol().broadcast_gossip(
            self.node.get_communication_protocol().build_msg(PeerRoundUpdatedCommand.get_name(),
                                                    round=self.node.get_local_state().get_experiment().round))

        await self.continue_p2p_round_initialization()

    async def on_enter_gossippingFullModel(self):
        """Gossip the model."""
        await GossipFullModelStage.execute(
            candidates=self.candidates,
            node=self.node,
        )

        await self.continue_p2p_round_initialization()

    async def on_enter_p2pVoting_startingVoting(self, *args, **kwargs):
        """Set the model initialized."""
        await self.continue_p2p_voting()

    async def on_enter_p2pVoting_voting(self):
        """Vote for the train set."""
        logger.info(self.node.address, "⏳ Voting for the train set.")
        await VoteTrainSetStage.execute(
            node=self.node,
        )

        await self.continue_p2p_voting()

    async def on_enter_p2pVoting_votingFinished(self, *args, **kwargs):
        """Finish the voting."""
        logger.info(self.node.address, "🤖 Voting finished.")

    async def on_final_p2p_voting(self, *args, **kwargs):
        """Finish the voting."""
        await AggregatingVoteTrainSetStage.execute(
            network_state=self.network_state,
            node=self.node,
        )

        await self.next_stage()

    async def on_enter_p2pLearning_evaluating(self):
        """Evaluate the model."""
        await EvaluateStage.execute(
            node=self.node,
        )
        await self.continue_p2p_learning()

    async def on_enter_p2pLearning_training(self):
        """Train the model."""
        await TrainStage.execute(
            node=self.node,
        )

        # Send aggregated model to the workflow
        await self.node.get_event_handler().aggregate(
            self.node.get_learner().get_P2PFLModel(),
            self.node.address
        )

        await self.continue_p2p_learning()

    async def on_enter_p2pLearning_gossippingPartialAggregation(self):
        """Gossip the partial model."""
        await GossipPartialModelStage.execute(
            network_state=self.network_state,
            candidates=self.candidates,
            node=self.node,
        )
        await self.continue_p2p_learning()

    async def on_enter_p2pLearning_aggregating(self):
        """Aggregate the models."""
        # Set aggregated model
        agg_model = self.node.get_aggregator().aggregate(self.network_state.get_all_models())
        self.node.get_learner().set_P2PFLModel(agg_model)

        await self.continue_p2p_learning()

    async def on_enter_p2pLearning_aggregationFinished(self):
        """Finish the aggregation."""
        logger.info(self.node.address, "🤖 Aggregation finished.")

    async def on_final_p2p_learning(self):
        """Finish the training."""
        await self.next_stage()

    async def on_enter_roundFinished(self):
        """Finish the round."""
        logger.info(
            self.node.address,
            f"🎉 Round {self.node.get_local_state().round} finished.",
        )
        await self.next_stage()

    async def on_enter_trainingFinished(self):
        """Finish the training."""
        await EvaluateStage.execute(
            node=self.node,
        )

        # Clean state
        self.node.get_local_state().clear()
        self.network_state.clear()

        logger.info(self.node.address, "😋 Training finished!!")

        await self.node.get_event_handler().training_finished()


    ##############
    # CONDITIONS #
    ##############
    def is_model_initialized(self, *args, **kwargs):
        """
        Check if the model has been initialized.

        This is done by checking if the learner's model round is greater than the local state round.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            bool: True if the model is initialized, False otherwise.

        """
        learner_round = self.node.get_learner().get_P2PFLModel().get_round()

        return learner_round == self.node.get_local_state().round if learner_round is not None else False

    def is_all_models_initialized(self, *args, **kwargs):
        """Check if all models have been initialized."""
        rounds = self.network_state.get_all_rounds()
        current_round = self.node.get_local_state().round
        initialized_nodes = sum(1 for value in rounds.values() if value == current_round)

        return initialized_nodes >= len(rounds)

    def candidate_exists(self, *args, **kwargs):
        """Check if there are candidates."""
        return len(self.candidates) > 0

    def in_train_set(self, *args, **kwargs):
        """Check if the node is in the train set."""
        return self.node.address in self.node.get_local_state().train_set

    def is_total_rounds_reached(self, *args, **kwargs):
        """Check if the total rounds have been reached."""
        return self.node.get_local_state().round >= self.node.get_local_state().total_rounds

    def is_all_models_received(self, *args, **kwargs):
        """Check if all models have been received."""
        return len(self.node.get_local_state().train_set) == len(self.network_state.get_all_models())


    ########################
    # CANDIDATES CALLBACKS #
    ########################
    def get_partial_gossipping_candidates(self):
        """Get the candidates from the train set to gossip the partial model."""
        def candidate_condition(node: str) -> set[str]:
            local_state = self.node.get_local_state()
            return set(local_state.train_set) - set(self.network_state.get_aggregation_sources(node) or [])

        candidates = set(self.node.get_local_state().train_set) - {self.node.address}
        self.candidates = [n for n in candidates if len(candidate_condition(n)) != 0]
        logger.debug(self.node.address, f"📡 Candidates to gossip to: {self.candidates}")

    def get_full_gossipping_candidates(self):
        """Get the candidates from the train set to gossip the full model."""
        fixed_round = self.node.get_local_state().round
        def candidate_condition(node: str) -> bool:
            return self.network_state.get_round(node) < fixed_round

        self.candidates = [n for n in self.node.communication_protocol.get_neighbors(only_direct=True) if candidate_condition(n)]
        logger.debug(self.node.local_state.address, f"📡 Candidates to gossip to: {self.candidates}")
