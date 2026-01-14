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
from p2pfl.learning.frameworks.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.stages.base_node import (
    AggregatingVoteTrainSetStage,
    EvaluateStage,
    GossipFullModelStage,
    GossipPartialModelStage,
    TrainStage,
    VoteTrainSetStage,
)
from p2pfl.stages.network_state.basic_network_state import BasicNetworkState
from p2pfl.stages.workflows.models.learning_workflow_model import LearningWorkflowModel
from p2pfl.utils.pytransitions import StateAdapter, TransitionAdapter

if TYPE_CHECKING:
    from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
    from p2pfl.node import Node
    from p2pfl.stages.local_state.dfl_node_state import DFLLocalNodeState


def get_states() -> list[dict]:
    """Define the states for the workflow."""
    states = [
        # Setup & initial synchronization
        StateAdapter(name="waitingSetup"),
        StateAdapter(name="startingTraining"),
        StateAdapter(name="waitingForSynchronization"),
        StateAdapter(name="nodesSynchronized"),
        StateAdapter(name="initializingInitiator"),
        StateAdapter(name="waitingForFullModel"),
        StateAdapter(name="updatingRound"),
        StateAdapter(name="gossipingFullModel"),
        StateAdapter(name="waitingForNetworkStart"),
        StateAdapter(name="roundInitialized"),
        StateAdapter(
            name="p2pVoting",
            initial="startingVoting",
            on_final="on_final_p2p_voting",
            children=[
                StateAdapter(name="startingVoting"),
                StateAdapter(name="voting"),
                StateAdapter(name="waitingVoting", timeout=Settings.training.VOTE_TIMEOUT, on_timeout="voting_timeout"),
                StateAdapter(name="votingFinished", final=True),
            ],
        ),
        StateAdapter(
            name="p2pLearning",
            initial="evaluating",
            children=[
                StateAdapter(name="evaluating"),
                StateAdapter(name="training"),
                StateAdapter(name="gossipingPartialAggregation"),
                StateAdapter(
                    name="waitingForPartialAggregation", timeout=Settings.training.AGGREGATION_TIMEOUT, on_timeout="aggregation_timeout"
                ),
                StateAdapter(name="aggregating"),
                StateAdapter(name="aggregationFinished", final=True),
            ],
        ),
        StateAdapter(name="roundFinished"),
        StateAdapter(name="trainingFinished", final=True),
    ]

    return [state.to_dict() for state in states]


def get_transitions() -> list[dict]:
    """Define the transitions for the workflow."""
    transitions = [
        # Setup & initial synchronization
        TransitionAdapter(trigger="setup", source="waitingSetup", dest="startingTraining"),
        TransitionAdapter(trigger="next_stage", source="startingTraining", dest="waitingForSynchronization"),
        TransitionAdapter(
            trigger="next_stage", source="waitingForSynchronization", dest="nodesSynchronized", conditions="is_all_nodes_started"
        ),
        # Model initialization
        TransitionAdapter(trigger="next_stage", source="nodesSynchronized", dest="initializingInitiator", conditions="is_initiator_node"),
        TransitionAdapter(trigger="next_stage", source="initializingInitiator", dest="updatingRound"),
        TransitionAdapter(trigger="next_stage", source="nodesSynchronized", dest="waitingForFullModel"),
        TransitionAdapter(trigger="next_stage", source="waitingForFullModel", dest="updatingRound", conditions="is_full_model_ready"),
        # Update round
        TransitionAdapter(
            trigger="next_stage",
            source="updatingRound",
            dest="gossipingFullModel",
            prepare=["get_full_gossiping_candidates"],
            conditions="candidate_exists",
        ),
        TransitionAdapter(trigger="next_stage", source="updatingRound", dest="waitingForNetworkStart"),
        # Gossip full model
        TransitionAdapter(
            trigger="next_stage", source="gossipingFullModel", dest="roundInitialized", conditions="is_all_models_initialized"
        ),
        TransitionAdapter(trigger="next_stage", source="gossipingFullModel", dest="waitingForNetworkStart"),
        TransitionAdapter(
            trigger="next_stage", source="waitingForNetworkStart", dest="roundInitialized", conditions="is_all_models_initialized"
        ),
        # Workflow finish check
        TransitionAdapter(trigger="next_stage", source="roundInitialized", dest="trainingFinished", conditions="is_total_rounds_reached"),
        TransitionAdapter(trigger="next_stage", source="roundInitialized", dest="p2pVoting"),
        # Voting
        TransitionAdapter(trigger="next_stage", source="p2pVoting_startingVoting", dest="p2pVoting_voting"),
        TransitionAdapter(trigger="next_stage", source="p2pVoting_voting", dest="p2pVoting_waitingVoting"),
        TransitionAdapter(
            trigger="next_stage", source="p2pVoting_waitingVoting", dest="p2pVoting_votingFinished", conditions="is_all_votes_received"
        ),
        TransitionAdapter(trigger="voting_timeout", source="p2pVoting_waitingVoting", dest="p2pVoting_votingFinished"),
        # Voting outcome
        TransitionAdapter(trigger="next_stage", source="p2pVoting", dest="p2pLearning", conditions="in_train_set"),
        TransitionAdapter(trigger="next_stage", source="p2pVoting", dest="waitingForFullModel"),
        # Learning
        TransitionAdapter(trigger="next_stage", source="p2pLearning_evaluating", dest="p2pLearning_training"),
        TransitionAdapter(
            trigger="next_stage",
            source="p2pLearning_training",
            dest="p2pLearning_gossipingPartialAggregation",
            prepare=["get_partial_gossiping_candidates"],
            conditions="candidate_exists",
        ),
        TransitionAdapter(trigger="next_stage", source="p2pLearning_training", dest="p2pLearning_waitingForPartialAggregation"),
        TransitionAdapter(
            trigger="next_stage", source="p2pLearning_gossipingPartialAggregation", dest="p2pLearning_waitingForPartialAggregation"
        ),
        TransitionAdapter(
            trigger="next_stage",
            source="p2pLearning_waitingForPartialAggregation",
            dest="p2pLearning_aggregating",
            conditions="is_all_models_received",
        ),
        TransitionAdapter(trigger="aggregation_timeout", source="p2pLearning_waitingForPartialAggregation", dest="p2pLearning_aggregating"),
        TransitionAdapter(trigger="next_stage", source="p2pLearning_aggregating", dest="p2pLearning_aggregationFinished"),
        # Loop
        TransitionAdapter(trigger="next_stage", source="p2pLearning", dest="roundFinished"),
        TransitionAdapter(trigger="next_stage", source="roundFinished", dest="updatingRound", conditions="is_all_models_received"),
        TransitionAdapter(trigger="next_stage", source="roundFinished", dest="waitingForFullModel"),
        # Event handler transitions
        TransitionAdapter(
            trigger="node_started",
            source=["waitingSetup", "startingTraining", "waitingForSynchronization"],
            dest=None,
            prepare="create_peer",
        ),
        TransitionAdapter(trigger="peer_round_updated", source="*", dest=None, prepare="save_peer_round_updated"),
        TransitionAdapter(trigger="full_model_received", source="*", dest=None, prepare="save_full_model"),
        TransitionAdapter(trigger="vote", source="*", dest=None, prepare="save_votes"),
        TransitionAdapter(trigger="aggregated_models_received", source="*", dest=None, prepare="save_aggregated_models"),
        TransitionAdapter(trigger="aggregate", source="*", dest=None, prepare="save_aggregation"),
    ]

    return [transition.to_dict() for transition in transitions]


class BasicLearningWorkflowModel(LearningWorkflowModel):
    """Model for the training workflow."""

    def __init__(self, node: Node, local_state: DFLLocalNodeState, network_state: BasicNetworkState) -> None:
        """Initialize the workflow model."""
        self.network_state: BasicNetworkState = network_state
        self.local_state: DFLLocalNodeState = local_state

        self.is_initiator: bool = False

        super().__init__(
            node=node,
        )

    async def is_finished(self) -> bool:
        """Check if the workflow has finished."""
        return self.state == "trainingFinished"

    ########################################
    # EVENTS (Overridden by pytransitions) #
    ########################################

    # Timeout events
    async def voting_timeout(self) -> bool:
        """Handle the voting timeout event."""
        raise RuntimeError("Should be overridden!")

    async def aggregation_timeout(self) -> bool:
        """Handle the aggregation timeout event."""
        raise RuntimeError("Should be overridden!")

    # Event handler events
    async def node_started(self, source: str) -> bool:
        """Handle the node started event."""
        raise RuntimeError("Should be overridden!")

    async def peer_round_updated(self, source: str, round: int) -> bool:
        """Handle the peer round updated event."""
        raise RuntimeError("Should be overridden!")

    async def full_model_received(self, source: str, round: int, weights: bytes) -> bool:
        """Handle the full model received event."""
        raise RuntimeError("Should be overridden!")

    async def vote(self, source: str, round: int, tmp_votes: list[tuple[str, int]]) -> bool:
        """Handle the vote event."""
        raise RuntimeError("Should be overridden!")

    async def aggregated_models_received(self, source: str, round: int, aggregated_models: list[str]) -> bool:
        """Handle the aggregated models received event."""
        raise RuntimeError("Should be overridden!")

    async def aggregate(self, model: P2PFLModel, source: str) -> bool:
        """Handle the aggregate event."""
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
        self.is_initiator = is_initiator

        # Set the experiment parameters
        self.local_state.set_experiment(experiment_name, rounds, epochs, trainset_size)
        learner.set_epochs(self.local_state.epochs)

        experiment = self.local_state.get_experiment()
        if experiment is None:
            raise ValueError("Experiment not initialized")
        exp_name = experiment.exp_name

        # Comunicate start learning command
        await communication_protocol.broadcast_gossip(
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

        # self._fire_next_stage("next_stage")

    async def on_enter_waitingForSynchronization(self):
        """Wait for the synchronization."""
        logger.info(self.node.address, "⏳ Waiting initialization.")

        communication_protocol = self.node.get_communication_protocol()

        # Wait and gossip model initialization
        try:
            await communication_protocol.broadcast_gossip(communication_protocol.build_msg(NodeInitializedCommand.get_name()))
        except Exception as e:
            logger.debug(self.node.address, f"Error broadcasting node initialization command: {e}")

        # Set self model initialized
        await self.node_started(
            source=self.node.address,
        )

    async def on_enter_nodesSynchronized(self, *args, **kwargs):
        """All nodes are synchronized."""
        logger.debug(self.node.address, "🤝 All nodes synchronized.")

    async def on_enter_initializingInitiator(self, *args, **kwargs):
        """Initialize the initiator node."""
        logger.info(self.node.address, "🚀 Initializing the network as initiator.")

        # Initialize round 0
        self.local_state.increase_round()

    async def on_enter_roundInitialized(self, *args, **kwargs):
        """Round initialized."""
        logger.debug(self.node.address, "🤖 Round initialized.")

    async def on_enter_updatingRound(self, *args, **kwargs):
        """Update the round."""
        # Set next round and reset variables
        self.network_state.reset_all_rounds()

        logger.info(
            self.node.address,
            f"🎉 Round {self.local_state.round} of {self.local_state.total_rounds} started.",
        )

        # Send event to the workflow
        await self.peer_round_updated(self.node.address, self.local_state.round)

        # Communicate round update
        await self.node.get_communication_protocol().broadcast_gossip(
            self.node.get_communication_protocol().build_msg(
                PeerRoundUpdatedCommand.get_name(), round=self.local_state.get_experiment().round
            )
        )

    async def on_enter_gossipingFullModel(self):
        """Gossip the model."""
        await GossipFullModelStage.execute(
            candidates=self.candidates,
            node=self.node,
        )

    async def on_enter_p2pVoting_startingVoting(self, *args, **kwargs):
        """Set the model initialized."""
        pass

    async def on_enter_p2pVoting_voting(self):
        """Vote for the train set."""
        logger.info(self.node.address, "⏳ Voting for the train set.")
        await VoteTrainSetStage.execute(
            node=self.node,
        )

    async def on_enter_p2pVoting_votingFinished(self, *args, **kwargs):
        """Finish the voting."""
        logger.info(self.node.address, "🤖 Voting finished.")

    async def on_final_p2p_voting(self, *args, **kwargs):
        """Finish the voting."""
        await AggregatingVoteTrainSetStage.execute(
            network_state=self.network_state,
            node=self.node,
        )

    async def on_enter_p2pLearning_evaluating(self):
        """Evaluate the model."""
        await EvaluateStage.execute(
            node=self.node,
        )

    async def on_enter_p2pLearning_training(self):
        """Train the model."""
        await TrainStage.execute(
            node=self.node,
        )

        # Send aggregated model to the workflow
        await self.aggregate(self.node.get_learner().get_model(), self.node.address)

    async def on_enter_p2pLearning_gossipingPartialAggregation(self):
        """Gossip the partial model."""
        await GossipPartialModelStage.execute(
            network_state=self.network_state,
            candidates=self.candidates,
            node=self.node,
        )

    async def on_enter_p2pLearning_aggregating(self):
        """Aggregate the models."""
        # Set aggregated model
        agg_model = self.node.get_aggregator().aggregate(self.network_state.get_all_models())
        self.node.get_learner().set_model(agg_model)

    async def on_enter_p2pLearning_aggregationFinished(self):
        """Finish the aggregation."""
        logger.info(self.node.address, "🤖 Aggregation finished.")

    async def on_enter_roundFinished(self):
        """Finish the round."""
        logger.info(
            self.node.address,
            f"🎉 Round {self.local_state.round} finished.",
        )

    async def on_enter_trainingFinished(self):
        """Finish the training."""
        await EvaluateStage.execute(
            node=self.node,
        )

        # Clean state
        self.local_state.clear()
        self.network_state.clear()

        logger.info(self.node.address, "😋 Training finished!!")

    ##############
    # CONDITIONS #
    ##############
    def is_initiator_node(self, *args, **kwargs):
        """
        Check if the node is the initiator.

        Returns:
            bool: True if the node is the initiator, False otherwise.

        """
        return self.is_initiator

    def is_full_model_ready(self, *args, **kwargs):
        """Check if the full model is ready."""
        return self.network_state.get_round(self.node.address) == self.local_state.round

    def is_all_models_initialized(self, *args, **kwargs):
        """Check if all models have been initialized."""
        rounds = self.network_state.get_all_rounds()
        current_round = self.local_state.round
        initialized_nodes = sum(1 for value in rounds.values() if value == current_round)

        return initialized_nodes >= len(rounds)

    def candidate_exists(self, *args, **kwargs):
        """Check if there are candidates."""
        return len(self.candidates) > 0

    def in_train_set(self, *args, **kwargs):
        """Check if the node is in the train set."""
        return self.node.address in self.local_state.train_set

    def is_total_rounds_reached(self, *args, **kwargs):
        """Check if the total rounds have been reached."""
        return self.local_state.round >= self.local_state.total_rounds

    def is_all_models_received(self, *args, **kwargs):
        """Check if all models have been received."""
        return len(self.local_state.train_set) == len(self.network_state.get_all_models())

    ########################
    # CANDIDATES CALLBACKS #
    ########################
    def get_partial_gossiping_candidates(self):
        """Get the candidates from the train set to gossip the partial model."""

        def candidate_condition(node: str) -> set[str]:
            return set(self.local_state.train_set) - set(self.network_state.get_aggregation_sources(node) or [])

        candidates = set(self.local_state.train_set) - {self.node.address}
        self.candidates = [n for n in candidates if len(candidate_condition(n)) != 0]
        logger.debug(self.node.address, f"📡 Candidates to gossip to: {self.candidates}")

    def get_full_gossiping_candidates(self):
        """Get the candidates from the train set to gossip the full model."""
        fixed_round = self.local_state.round

        def candidate_condition(node: str) -> bool:
            return self.network_state.get_round(node) < fixed_round

        self.candidates = [n for n in self.node.communication_protocol.get_neighbors(only_direct=True) if candidate_condition(n)]
        logger.debug(self.local_state.address, f"📡 Candidates to gossip to: {self.candidates}")

    ########################
    # EVENT HANDLER EVENTS #
    ########################
    async def send_network_ready(self, *args, **kwargs):
        """Send the network ready event."""
        logger.info(self.node.address, "✅ Network ready.")
        await self.network_ready()

    async def send_models_ready(self, *args, **kwargs):
        """Send the models updated event."""
        logger.info(self.node.address, "✅ Models ready.")
        await self.models_ready()

    async def send_votes_ready(self, *args, **kwargs):
        """Send the votes ready event."""
        logger.info(self.node.address, "✅ Votes ready.")
        await self.votes_ready()

    async def send_aggregation_ready(self, *args, **kwargs):
        """Send the aggregation ready event."""
        logger.info(self.node.address, "✅ Aggregation ready.")
        await self.aggregation_ready()

    async def send_full_model_ready(self, *args, **kwargs):
        """Send the full model ready event."""
        logger.info(self.node.address, "✅ Full model ready.")
        await self.full_model_ready()

    async def send_peers_ready(self, *args, **kwargs):
        """Send the peers ready event."""
        logger.info(self.node.address, "✅ Peers ready.")
        await self.peers_ready()

    ###########################
    # EVENT HANDLER CALLBACKS #
    ###########################
    async def on_enter_waitingContextUpdate(self, *args, **kwargs):
        """Wait for the context to be updated."""
        logger.info(self.node.address, "⏳ Waiting for the context to be updated.")

    #########################
    # EVENT HANDLER SETTERS #
    #########################
    async def save_started_node(self, source: str):
        """Save the votes."""
        try:
            self.network_state.update_round(source, -1)
            logger.debug(self.node.address, f"📦 Node started: {source}")
        except Exception as e:
            logger.error(self.node.address, f"❌ Error saving started node {source}: {e}")

    async def save_votes(self, source: str, round: int, tmp_votes: list[tuple[str, int]]):
        """Save the votes."""
        if self.local_state is None:
            logger.error(self.node.address, f"Local state is None, cannot update votes for {source}")
            return

        if round == self.local_state.round:
            for train_set_id, vote in tmp_votes:
                self.network_state.add_vote(source, train_set_id, vote)
            logger.debug(self.node.address, f"📦 Votes received from {source}: {tmp_votes}")
        else:
            logger.error(self.node.address, f"📦 Votes not received from {source}: {tmp_votes} (expected {self.local_state.round})")

    async def save_aggregation(self, model: P2PFLModel, source: str):
        """Save the aggregation."""
        try:
            self.network_state.add_model(model, source)
            logger.debug(self.node.address, f"📦 Model received from {source}: {model}")
        except Exception as e:
            logger.error(self.node.address, f"❌ Error saving model from {source}: {e}")

    async def create_peer(self, source: str):
        """Update the peer round."""
        try:
            self.network_state.add_peer(source)
            logger.debug(self.node.address, f"📡 {source} peer created")
        except Exception as e:
            logger.error(self.node.address, f"❌ Error creating peer {source}: {e}")

    async def save_peer_round_updated(
        self,
        source: str,
        round: int,
    ):
        """Initialize model."""
        local_round = self.local_state.round
        if local_round is None:
            logger.error(self.node.address, f"Local state is None, cannot update round for {source}")
            return

        if round in [local_round, local_round + 1]:
            self.network_state.update_round(source, round)
            logger.debug(self.node.address, f"📡 Peer round updated: {source} -> {round}")
        else:
            logger.error(self.node.address, f"📡 Peer round not updated: {source} -> {round} (local round: {local_round})")

    async def save_full_model(self, source: str, round: int, weights: bytes):
        """Initialize model."""
        # Check source
        logger.info(self.node.address, "📦 Full model received.")

        try:
            if round != self.local_state.round + 1:
                logger.warning(
                    self.node.address, f"⚠️ Full model round {round} does not match local round {self.local_state.round+1}. Ignoring."
                )
                return

            # Set new weights and increase round
            self.node.get_learner().set_model(weights)
            self.local_state.increase_round()

            logger.info(self.node.address, "🤖 Model Weights Initialized")

        except DecodingParamsError:
            logger.error(self.node.address, "❌ Error decoding parameters.")
        except ModelNotMatchingError:
            logger.error(self.node.address, "❌ Models not matching.")
        except Exception as e:
            logger.error(self.node.address, f"❌ Unknown error adding model: {e}")

    async def save_aggregated_models(self, source: str, round: int, aggregated_models: list[str]):
        """Save the aggregated models."""
        if round == self.local_state.round:
            for aggregated_model in list(aggregated_models):
                self.network_state.add_aggregated_from(source, aggregated_model)

            logger.debug(self.node.address, f"📦 Aggregated models received from {source}: {aggregated_models}")
        else:
            logger.error(
                self.node.address,
                f"📦 Aggregated models not received from {source}: {aggregated_models}\
                         (expected {self.local_state.round})",
            )

    ############################
    # EVENT HANDLER CONDITIONS #
    ############################
    def is_all_nodes_started(self, *args, **kwargs):
        """Check if all nodes have started."""
        return len(self.network_state.list_peers()) == (len(self.node.communication_protocol.get_neighbors(only_direct=False)) + 1)

    def is_all_votes_received(self, *args, **kwargs):
        """Check if all votes from neis have been received."""
        neighbors = self.network_state.list_peers()
        votes = self.network_state.get_all_votes()

        # Check if all neighbors have voted
        return all(votes.get(nei) for nei in neighbors)
