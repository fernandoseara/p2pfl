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
from typing import TYPE_CHECKING

from p2pfl.learning.frameworks.custom_model_factory import CustomModelFactory
from p2pfl.learning.frameworks.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.management.logger import logger
from p2pfl.utils.pytransitions import StateAdapter, TimeoutMachine, TransitionAdapter
from p2pfl.workflow import on_message
from p2pfl.workflow.async_dfl.stages.compute_priority_stage import ComputePriorityStage
from p2pfl.workflow.async_dfl.stages.gossip_model_stage import GossipModelStage
from p2pfl.workflow.async_dfl.stages.select_neighbor_stage import SelectNeighborsStage
from p2pfl.workflow.basic_dfl.stages.evaluate_stage import EvaluateStage
from p2pfl.workflow.engine.workflow import Workflow
from p2pfl.workflow.factory import WorkflowType

if TYPE_CHECKING:
    from transitions import Machine

    from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
    from p2pfl.node import Node


@dataclass
class AsyncPeerState:
    """Per-peer state for AsyncDFL workflow."""

    round_number: int = 0
    push_sum_weight: float = 1.0
    model: P2PFLModel | None = None
    losses: list[float] = field(default_factory=list)
    push_time: int = 0
    mixing_weight: float = 1.0
    p2p_updating_idx: int = 0

    def add_loss(self, round: int, loss: float) -> None:
        """Record a loss value at the given round index."""
        while len(self.losses) <= round:
            self.losses.append(0.0)
        self.losses[round] = loss


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
        TransitionAdapter(trigger="stop_learning", source="*", dest="trainingFinished", before="_before_stop_learning"),
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


class AsyncDFL(Workflow):
    """Model for the training workflow."""

    # ══════════════════════════════════════════════
    #  Initialization
    # ══════════════════════════════════════════════

    def __init__(self, node: Node) -> None:
        """Initialize the workflow model."""
        # Workflow state
        self.peers: dict[str, AsyncPeerState] = {}

        self.candidates: list[str] = []
        self.tau: int = 2  # τ - network update interval

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

        # Wrap the model for async FL
        model = node.learner.get_model()
        node.learner.set_model(CustomModelFactory.create_model("AsyDFL", model))

    @property
    def workflow_type(self) -> WorkflowType:
        """Return the workflow type."""
        return WorkflowType.ASYNC

    def _cleanup(self) -> None:
        """Clean up subclass state and remove from state machine."""
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

    async def on_enter_startingTraining(self, *args, **kwargs):
        """Start the training."""
        logger.info(self.node.address, "⏳ Starting training.")

    async def on_enter_waitingForSynchronization(self, *args, **kwargs):
        """Wait for the synchronization."""
        logger.info(self.node.address, "⏳ Waiting initialization.")

        communication_protocol = self.node.communication_protocol

        await EvaluateStage.execute(
            node=self.node,
        )

        # Communicate Initialization
        await communication_protocol.broadcast(communication_protocol.build_msg("node_initialized"))

        # Set self model initialized
        await self.node_started(
            source=self.node.address,
        )

    async def create_peer(self, source: str):
        """Create a new peer entry."""
        try:
            if source in self.peers:
                raise ValueError(f"Address {source} already exists in peers")
            self.peers[source] = AsyncPeerState()
            logger.debug(self.node.address, f"📡 {source} peer created")
        except Exception as e:
            logger.error(self.node.address, f"❌ Error creating peer {source}: {e}")

    def is_all_nodes_started(self, *args, **kwargs):
        """Check if all nodes have started."""
        return len(self.peers) == (len(self.node.communication_protocol.get_neighbors(only_direct=True)) + 1)

    async def send_network_ready(self, *args, **kwargs):
        """Send the network ready event."""
        logger.info(self.node.address, "✅ Network ready.")
        await self.network_ready()

    async def on_enter_nodesSynchronized(self, *args, **kwargs):
        """Nodes synchronized."""
        communication_protocol = self.node.communication_protocol
        neighbors = list(communication_protocol.get_neighbors(only_direct=True).keys())

        # Log participants
        participants = neighbors + [self.node.address]
        logger.info(self.node.address, f"👥 Peers in network: {participants}")

        # Set mixing weights pt_j,i(0)
        weight = 1.0 / len(neighbors) if neighbors else 1.0
        for neighbor in neighbors:
            if neighbor in self.peers:
                self.peers[neighbor].mixing_weight = weight

    # ══════════════════════════════════════════════
    #  Phase 2: Training Round
    # ══════════════════════════════════════════════

    async def on_enter_trainingRound_debiasingModel(self, *args, **kwargs):
        """De-bias the model (Equation 3)."""
        logger.debug(self.node.address, "🤖 Debiasing model.")

        model = self.node.learner.get_model()
        peer = self.peers.get(self.node.address)
        if peer is not None:
            model.set_push_sum_weight(peer.push_sum_weight)

    async def on_enter_trainingRound_updatingLocalModel(self, *args, **kwargs):
        """Update the local model with a training batch."""
        logger.info(self.node.address, "🏋️‍♀️ Updating local model...")
        await self.node.learner.train_on_batch()
        self.peers[self.node.address].model = self.node.learner.get_model()

    async def on_enter_trainingRound_sendingTrainingLoss(self, *args, **kwargs):
        """Broadcast the training loss to peers."""
        communication_protocol = self.node.communication_protocol

        training_loss = self.node.learner.get_model().last_training_loss
        self.peers[self.node.address].add_loss(self.round, training_loss)

        logger.info(self.node.address, "📢 Broadcasting loss values.")
        flattened_loss = [str(training_loss)]
        try:
            await communication_protocol.broadcast(
                communication_protocol.build_msg(
                    "loss_information_updating",
                    flattened_loss,
                    round=self.round,
                )
            )
        except Exception as e:
            logger.warning(self.node.address, f"⚠️ Failed to broadcast loss: {e}")

    async def save_loss_information(self, source: str, round: int, loss: float):
        """Save the loss information."""
        try:
            self.peers[source].add_loss(round, loss)
            logger.debug(self.node.address, f"📡 {source} loss updated to {loss} for round {round}")
        except Exception as e:
            logger.error(self.node.address, f"❌ Error saving loss information from {source} for round {round}: {e}")

    def check_iteration_network_updating(self, *args, **kwargs):
        """Check if it is time to update the model with network updating (Line 7)."""
        return self.round % self.tau == 0

    # ══════════════════════════════════════════════
    #  Phase 3: Network Updating
    # ══════════════════════════════════════════════

    async def get_gossip_candidates(self):
        """Get the candidates from the train set to gossip the partial model."""
        # Compute the candidates priorities
        neighbor_priorities = await ComputePriorityStage.execute(  # TODO: ESTO NO TIENE MUCHO SENTIDO... PARA ESO SE USA UNA FUNCION ...
            peers=self.peers,
            node=self.node,
        )
        logger.info(self.node.address, f"👥 Neighbor priorities: {neighbor_priorities}")

        # Select the candidates
        self.candidates = await SelectNeighborsStage.execute(
            neighbor_priorities=neighbor_priorities,
        )

        logger.info(self.node.address, f"🧾 Selected neighbors: {self.candidates}")

    async def on_enter_trainingRound_networkUpdating_gossipingModel(self, *args, **kwargs):
        """Gossip the model to selected neighbors."""
        logger.info(self.node.address, "📡 Gossiping model...")
        await GossipModelStage.execute(
            peers=self.peers,
            candidates=self.candidates,
            node=self.node,
        )

    async def save_model(self, source: str, round: int, weights: bytes, num_samples: int, contributors: list[str]):
        """Save a received model from a peer."""
        logger.info(self.node.address, "📦 Model received.")

        try:
            model = self.node.learner.get_model().build_copy(params=weights, num_samples=num_samples, contributors=contributors)
            self.peers[source].model = model

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
            self.peers[source].round_number = index
            logger.debug(self.node.address, f"📡 {source} round updated to {index}")
        except Exception as e:
            logger.error(self.node.address, f"❌ Error saving iteration index from {source}: {e}")

    async def save_push_sum_weight(self, source: str, push_sum_weight: float):
        """Save the push sum weight."""
        try:
            self.peers[source].push_sum_weight = push_sum_weight
            logger.debug(self.node.address, f"📡 {source} push sum weight updated to {push_sum_weight}")
        except Exception as e:
            logger.error(self.node.address, f"❌ Error saving push sum weight from {source}: {e}")

    async def on_enter_trainingRound_networkUpdating_aggregating(self):
        """Aggregate models from peers (push-sum)."""
        communication_protocol = self.node.communication_protocol

        logger.info(self.node.address, "🤝 Aggregating models...")

        self_peer = self.peers.get(self.node.address)
        push_sum_weight = self_peer.push_sum_weight if self_peer else 1.0

        for neighbor, peer in self.peers.items():
            if neighbor == self.node.address:
                continue

            # Update push-sum weights (Equation 5)
            push_sum_weight += peer.mixing_weight * peer.push_sum_weight
            logger.debug(self.node.address, f"📡 {neighbor} push-sum weight updated to {push_sum_weight}")

            # Update index of P2P updating
            peer.p2p_updating_idx = self.round or 0

            # Send index of local iteration to neighbors
            try:
                await communication_protocol.send(
                    nei=neighbor,
                    msg=communication_protocol.build_msg(
                        "index_information_updating",
                        round=self.round,
                    ),
                )
            except Exception as e:
                logger.warning(self.node.address, f"⚠️ Failed to send iteration index to {neighbor}: {e}")

        # P2P updating of the model
        agg_model = self.node.aggregator.aggregate([p.model for p in self.peers.values() if p.model is not None])
        self.node.learner.set_model(agg_model)

        # Evaluate the model
        await EvaluateStage.execute(
            node=self.node,
        )

    async def on_enter_trainingRound_networkUpdating_networkUpdatingFinishing(self):
        """Finish the network updating."""
        logger.info(self.node.address, "🏁 Network updating finished.")

    # ══════════════════════════════════════════════
    #  Phase 4: Round Finish & Loop
    # ══════════════════════════════════════════════

    async def on_enter_trainingRound_roundFinishing(self):
        """Finish the round and advance the counter."""
        logger.info(
            self.node.address,
            f"🎉 Round {self.round} finished.",
        )

        self.increase_round()
        for p in self.peers.values():
            p.model = None

    def check_total_rounds_reached(self, *args, **kwargs):
        """Check if the total rounds have been reached (Line 2)."""
        if self.experiment is None or self.experiment.total_rounds is None:
            return False
        return self.round >= self.experiment.total_rounds

    # ══════════════════════════════════════════════
    #  Phase 5: Finish
    # ══════════════════════════════════════════════

    async def on_enter_trainingFinished(self):
        """Perform final evaluation and mark workflow as finished."""
        await EvaluateStage.execute(
            node=self.node,
        )
        self.mark_finished()
        logger.info(self.node.address, "Training finished!!")

    async def _before_stop_learning(self) -> None:
        """Stop learning and interrupt the current training."""
        logger.info(self.node.address, "Stopping learning")
        await self.node.learner.interrupt_fit()
        logger.experiment_finished(self.node.address)

    # ══════════════════════════════════════════════
    #  Message Handlers (@on_message)
    # ══════════════════════════════════════════════

    @on_message("node_initialized")
    async def handle_node_initialized(self, source: str, round: int, *args) -> None:
        """Handle node_initialized message."""
        await self.node_started(source)

    @on_message("loss_information_updating")
    async def handle_loss_information_updating(self, source: str, round: int, *args) -> None:
        """Handle loss_information_updating message."""
        if not args:
            raise ValueError("Loss is required")
        loss = float(args[0])
        await self.loss_information_received(source, round, loss)

    @on_message("index_information_updating")
    async def handle_index_information_updating(self, source: str, round: int, *args) -> None:
        """Handle index_information_updating message."""
        await self.iteration_index_received(source, index=round)

    @on_message("model_information_updating", weights=True)
    async def handle_model_information_updating(
        self, source: str, round: int, weights: bytes, contributors: list[str] | None, num_samples: int | None
    ) -> None:
        """Handle model_information_updating weight message."""
        if contributors is None or num_samples is None:
            raise ValueError("Contributors and num_samples are required")
        await self.model_received(source, round, weights, num_samples, list(contributors))

    @on_message("push_sum_weight_information_updating")
    async def handle_push_sum_weight_information_updating(self, source: str, round: int, *args) -> None:
        """Handle push_sum_weight_information_updating message."""
        if not args:
            raise ValueError("Push-sum weight is required")
        push_sum_weight = float(args[0])
        await self.push_sum_weight_received(source, push_sum_weight=push_sum_weight)

    @on_message("pre_send_model")
    async def handle_pre_send_model(self, source: str, round: int, *args) -> str:
        """Handle pre_send_model message - check if we want to receive a model."""
        if not args:
            return "false"

        contributors = list(args[1:]) if len(args) > 1 else []

        # Accept if any contributor is new
        existing: set[str] = set()
        for p in self.peers.values():
            if p.model:
                existing.update(p.model.get_contributors())
        new_contributors = set(contributors) - existing
        if new_contributors:
            return "true"
        return "false"

    # ══════════════════════════════════════════════
    #  Pytransitions Runtime Stubs
    #  (these methods are injected by pytransitions
    #   at runtime — stubs exist for type checking)
    # ══════════════════════════════════════════════

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

    async def network_ready(self) -> bool:
        """Handle the network ready event."""
        raise RuntimeError("Should be overridden!")
