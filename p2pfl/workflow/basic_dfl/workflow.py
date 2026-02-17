#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
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
"""Workflows."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from p2pfl.learning.frameworks.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.utils.pytransitions import StateAdapter, TimeoutMachine, TransitionAdapter
from p2pfl.workflow import on_message
from p2pfl.workflow.basic_dfl.stages import (
    AggregatingVoteTrainSetStage,
    EvaluateStage,
    GossipFullModelStage,
    GossipPartialModelStage,
    TrainStage,
    VoteTrainSetStage,
)
from p2pfl.workflow.engine.workflow import Workflow
from p2pfl.workflow.factory import WorkflowType

if TYPE_CHECKING:
    from transitions import Machine

    from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
    from p2pfl.node import Node


@dataclass
class BasicPeerState:
    """Per-peer state for BasicDFL workflow."""

    round_number: int = 0
    model: P2PFLModel | None = None
    aggregated_from: list[str] = field(default_factory=list)
    votes: dict[str, int] = field(default_factory=dict)

    def reset_round(self) -> None:
        """Reset per-round mutable state."""
        self.model = None
        self.aggregated_from.clear()
        self.votes.clear()


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
        # Stop training
        TransitionAdapter(trigger="stop_learning", source="*", dest="trainingFinished"),
    ]

    return [transition.to_dict() for transition in transitions]


class BasicDFL(Workflow):
    """Model for the training workflow."""

    # ══════════════════════════════════════════════
    #  Initialization
    # ══════════════════════════════════════════════

    def __init__(self, node: Node) -> None:
        """Initialize the workflow model."""
        # Workflow state
        self.peers: dict[str, BasicPeerState] = {}
        self.train_set: list[str] = []
        self.candidates: list[str] = []

        # Create the state machine
        self._machine: Machine = TimeoutMachine(
            model=None,
            states=get_states(),
            transitions=get_transitions(),
            initial=None,
            queued="model",
            ignore_invalid_triggers=True,
            model_override=True,
        )

        super().__init__(node=node)

        # Add self to the machine
        self._machine.add_model(self, initial="waitingSetup")

    @property
    def workflow_type(self) -> WorkflowType:
        """Return the workflow type."""
        return WorkflowType.BASIC

    def _cleanup(self) -> None:
        """Clean up subclass state and remove from state machine."""
        self.train_set = []
        self.peers.clear()
        super()._cleanup()
        self._machine.remove_model(self)

    async def _run(self) -> None:
        """Run the learning loop."""
        await self.setup()
        while self.state != "trainingFinished":
            await self.next_stage()
            await asyncio.sleep(0.1)

    # ══════════════════════════════════════════════
    #  Phase 1: Setup & Synchronization
    # ══════════════════════════════════════════════

    async def on_enter_startingTraining(self, *args: Any, **kwargs: Any):
        """Start the training."""
        logger.info(self.node.address, "⏳ Starting training.")

    async def on_enter_waitingForSynchronization(self):
        """Wait for the synchronization."""
        logger.info(self.node.address, "⏳ Waiting initialization.")

        communication_protocol = self.node.communication_protocol

        # Wait and gossip model initialization
        try:
            await communication_protocol.broadcast_gossip(communication_protocol.build_msg("node_initialized"))
        except Exception as e:
            logger.debug(self.node.address, f"Error broadcasting node initialization command: {e}")

        # Set self model initialized
        await self.create_peer(
            source=self.node.address,
        )

    async def create_peer(self, source: str):
        """Register a new peer."""
        if source in self.peers:
            logger.error(self.node.address, f"❌ Peer {source} already exists")
            return
        self.peers[source] = BasicPeerState()
        logger.debug(self.node.address, f"📡 {source} peer created")

    def is_all_nodes_started(self, *args, **kwargs):
        """Check if all nodes have started."""
        return len(self.peers) == (len(self.node.communication_protocol.get_neighbors(only_direct=False)) + 1)

    async def on_enter_nodesSynchronized(self, *args, **kwargs):
        """All nodes are synchronized."""
        logger.debug(self.node.address, "🤝 All nodes synchronized.")

    # ══════════════════════════════════════════════
    #  Phase 2: Round Initialization
    # ══════════════════════════════════════════════

    async def on_enter_initializingInitiator(self, *args, **kwargs):
        """Initialize the initiator node."""
        logger.info(self.node.address, "🚀 Initializing the network as initiator.")

        # Initialize round 0
        self.increase_round()

    def is_initiator_node(self, *args, **kwargs):
        """Pytransitions condition: check if this node is the initiator."""
        return self.experiment is not None and self.experiment.is_initiator

    async def on_enter_updatingRound(self, *args, **kwargs):
        """Update the round."""
        # Set next round and reset variables
        for p in self.peers.values():
            p.reset_round()

        logger.info(
            self.node.address,
            f"🎉 Round {self.round} of {self.experiment.total_rounds} started.",
        )

        # Save own round update
        await self.save_peer_round_updated(self.node.address, self.round)

        # Communicate round update
        await self.node.communication_protocol.broadcast_gossip(
            self.node.communication_protocol.build_msg("peer_round_updated", round=self.round)
        )

    async def save_peer_round_updated(
        self,
        source: str,
        round: int,
    ):
        """Save a peer round update."""
        local_model_round = self.round

        if round in [local_model_round, local_model_round + 1]:
            self.peers[source].round_number = round
            logger.debug(self.node.address, f"📡 Peer round updated: {source} -> {round}")
        else:
            logger.error(self.node.address, f"📡 Peer round not updated: {source} -> {round} (local round: {local_model_round})")

    def get_full_gossiping_candidates(self):
        """Get the candidates from the train set to gossip the full model."""
        fixed_round = self.round

        def candidate_condition(node: str) -> bool:
            peer = self.peers.get(node)
            return peer is not None and peer.round_number < fixed_round

        self.candidates = [n for n in self.node.communication_protocol.get_neighbors(only_direct=True) if candidate_condition(n)]
        logger.debug(self.node.address, f"📡 Candidates to gossip to: {self.candidates}")

    async def on_enter_gossipingFullModel(self):
        """Gossip the model."""
        await GossipFullModelStage.execute(
            candidates=self.candidates,
            node=self.node,
        )

    def candidate_exists(self, *args, **kwargs):
        """Check if there are candidates."""
        return len(self.candidates) > 0

    def is_all_models_initialized(self, *args, **kwargs):
        """Check if all models have been initialized."""
        return all(p.round_number == self.round for p in self.peers.values())

    async def on_enter_roundInitialized(self, *args, **kwargs):
        """Round initialized."""
        logger.debug(self.node.address, "🤖 Round initialized.")

    def is_total_rounds_reached(self, *args, **kwargs):
        """Check if the total rounds have been reached."""
        if self.experiment is None or self.experiment.total_rounds is None:
            return False
        return self.round >= self.experiment.total_rounds

    # ══════════════════════════════════════════════
    #  Phase 3: Voting
    # ══════════════════════════════════════════════

    async def on_enter_p2pVoting_voting(self):
        """Vote for the train set."""
        logger.info(self.node.address, "⏳ Voting for the train set.")
        await VoteTrainSetStage.execute(
            node=self.node,
        )

    async def save_votes(self, source: str, round: int, tmp_votes: list[tuple[str, int]]):
        """Save the votes."""
        if round == self.round:
            peer = self.peers[source]
            for train_set_id, vote in tmp_votes:
                peer.votes[train_set_id] = peer.votes.get(train_set_id, 0) + vote
            logger.debug(self.node.address, f"📦 Votes received from {source}: {tmp_votes}")
        else:
            logger.error(self.node.address, f"📦 Votes not received from {source}: {tmp_votes} (expected {self.round})")

    def is_all_votes_received(self, *args, **kwargs):
        """Check if all votes from neis have been received."""
        return all(p.votes for p in self.peers.values())

    async def on_enter_p2pVoting_votingFinished(self, *args, **kwargs):
        """Finish the voting."""
        logger.info(self.node.address, "🤖 Voting finished.")

    async def on_final_p2p_voting(self, *args, **kwargs):
        """Aggregate votes and determine the train set."""
        await AggregatingVoteTrainSetStage.execute(
            peers=self.peers,
            node=self.node,
        )

    def in_train_set(self, *args, **kwargs):
        """Check if the node is in the train set."""
        return self.node.address in self.train_set

    # ══════════════════════════════════════════════
    #  Phase 4: Learning & Aggregation
    # ══════════════════════════════════════════════

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
        await self.aggregate(self.node.learner.get_P2PFLModel(), self.node.address)

    async def save_aggregation(self, model: P2PFLModel, source: str):
        """Save the aggregation."""
        try:
            self.peers[source].model = model
            logger.debug(self.node.address, f"📦 Model received from {source}: {model}")
        except KeyError:
            logger.error(self.node.address, f"❌ Unknown peer {source}")

    def get_partial_gossiping_candidates(self):
        """Get the candidates from the train set to gossip the partial model."""
        train_set = set(self.train_set)

        def candidate_condition(node: str) -> set[str]:
            peer = self.peers.get(node)
            aggregated_from = set(peer.aggregated_from) if peer else set()
            return train_set - aggregated_from

        candidates = train_set - {self.node.address}
        self.candidates = [n for n in candidates if len(candidate_condition(n)) != 0]
        logger.debug(self.node.address, f"📡 Candidates to gossip to: {self.candidates}")

    async def on_enter_p2pLearning_gossipingPartialAggregation(self):
        """Gossip the partial model."""
        await GossipPartialModelStage.execute(
            peers=self.peers,
            candidates=self.candidates,
            node=self.node,
        )

    async def save_aggregated_models(self, source: str, round: int, aggregated_models: list[str]):
        """Save the aggregated models."""
        if round == self.round:
            self.peers[source].aggregated_from.extend(aggregated_models)

            logger.debug(self.node.address, f"📦 Aggregated models received from {source}: {aggregated_models}")
        else:
            logger.error(
                self.node.address,
                f"📦 Aggregated models not received from {source}: {aggregated_models} (expected {self.round})",
            )

    def is_all_models_received(self, *args, **kwargs):
        """Check if all models have been received."""
        return len(self.train_set) == sum(1 for p in self.peers.values() if p.model is not None)

    async def on_enter_p2pLearning_aggregating(self):
        """Aggregate the models."""
        # Set aggregated model
        agg_model = self.node.aggregator.aggregate([p.model for p in self.peers.values() if p.model is not None])
        self.node.learner.set_P2PFLModel(agg_model)
        # Increase round
        self.increase_round()

    async def on_enter_p2pLearning_aggregationFinished(self):
        """Finish the aggregation."""
        logger.info(self.node.address, "🤖 Aggregation finished.")

    async def on_enter_roundFinished(self):
        """Finish the round."""
        logger.info(
            self.node.address,
            f"🎉 Round {self.round} finished.",
        )

    # ══════════════════════════════════════════════
    #  Phase 5: Finish & Full Model Reception
    # ══════════════════════════════════════════════

    async def on_enter_trainingFinished(self):
        """Finish the training."""
        await EvaluateStage.execute(
            node=self.node,
        )
        self.mark_finished()
        logger.info(self.node.address, "Training finished!!")

    async def save_full_model(self, source: str, round: int, weights: bytes):
        """Save a received full model and advance the round."""
        # Check source
        logger.info(self.node.address, "📦 Full model received.")

        try:
            local_round = self.round
            if round != local_round + 1:
                logger.warning(self.node.address, f"⚠️ Full model round {round} does not match local round {local_round}. Ignoring.")
                return

            # Set new weights and increase round
            self.node.learner.set_model(weights)
            self.increase_round()

            logger.info(self.node.address, "🤖 Model Weights Initialized")

        except DecodingParamsError:
            logger.error(self.node.address, "❌ Error decoding parameters.")
        except ModelNotMatchingError:
            logger.error(self.node.address, "❌ Models not matching.")
        except Exception as e:
            logger.error(self.node.address, f"❌ Unknown error adding model: {e}")

    def is_full_model_ready(self, *args, **kwargs):
        """Check if the full model is ready."""
        peer = self.peers.get(self.node.address)
        if peer is None:
            return False
        return peer.round_number < self.round

    async def save_started_node(self, source: str):
        """Mark a node as started."""
        try:
            self.peers[source].round_number = -1
            logger.debug(self.node.address, f"📦 Node started: {source}")
        except KeyError:
            logger.error(self.node.address, f"❌ Unknown peer {source}")

    # ══════════════════════════════════════════════
    #  Message Handlers (@on_message)
    # ══════════════════════════════════════════════

    @on_message("node_initialized")
    async def handle_node_initialized(self, source: str, round: int, *args) -> None:
        """Handle node_initialized message."""
        await self.node_started(source)

    @on_message("peer_round_updated")
    async def handle_peer_round_updated(self, source: str, round: int, *args) -> None:
        """Handle peer_round_updated message."""
        await self.peer_round_updated(source, round)

    @on_message("vote_train_set")
    async def handle_vote_train_set(self, source: str, round: int, *args) -> None:
        """Handle vote_train_set message."""
        if len(args) % 2 != 0:
            raise ValueError("Votes list must contain an even number of elements (peer, weight pairs).")
        votes = [(args[i], int(args[i + 1])) for i in range(0, len(args), 2)]
        await self.vote(source, round, votes)

    @on_message("models_aggregated")
    async def handle_models_aggregated(self, source: str, round: int, *args) -> None:
        """Handle models_aggregated message."""
        await self.aggregated_models_received(source, round, list(args))

    @on_message("pre_send_model")
    async def handle_pre_send_model(self, source: str, round: int, *args) -> str:
        """Handle pre_send_model message - check if we want to receive a model."""
        if not args:
            return "false"

        weight_command = args[0]
        contributors = list(args[1:]) if len(args) > 1 else []

        if weight_command == "add_model":
            if round > self.round:
                return "true"
            return "false"

        elif weight_command == "partial_model":
            existing: set[str] = set()
            for p in self.peers.values():
                if p.model:
                    existing.update(p.model.get_contributors())
            new_contributors = set(contributors) - existing
            if new_contributors:
                return "true"
            return "false"

        return "true"

    @on_message("partial_model", weights=True)
    async def handle_partial_model(
        self, source: str, round: int, weights: bytes, contributors: list[str] | None, num_samples: int | None
    ) -> None:
        """Handle partial_model weight message."""
        from transitions import MachineError

        if contributors is None or num_samples is None:
            raise ValueError("Contributors and num_samples are required")

        try:
            model = self.node.learner.get_model().build_copy(
                params=weights,
                num_samples=num_samples,
                contributors=list(contributors),
            )
            await self.aggregate(model, source)
        except MachineError:
            logger.debug(self.node.address, "Invalid state.")
        except DecodingParamsError:
            logger.error(self.node.address, "Error decoding parameters.")
        except ModelNotMatchingError:
            logger.error(self.node.address, "Models not matching.")
        except Exception as e:
            logger.error(self.node.address, f"Unknown error adding model: {e}")

    @on_message("add_model", weights=True)
    async def handle_add_model(
        self, source: str, round: int, weights: bytes, contributors: list[str] | None, num_samples: int | None
    ) -> None:
        """Handle add_model (full model) weight message."""
        await self.full_model_received(source, round, weights)

    # ══════════════════════════════════════════════
    #  Pytransitions Runtime Stubs
    #  (these methods are injected by pytransitions
    #   at runtime — stubs exist for type checking)
    # ══════════════════════════════════════════════

    async def voting_timeout(self) -> bool:
        """Handle the voting timeout event."""
        raise RuntimeError("Should be overridden!")

    async def aggregation_timeout(self) -> bool:
        """Handle the aggregation timeout event."""
        raise RuntimeError("Should be overridden!")

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
