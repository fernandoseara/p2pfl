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
"""Model for the basic DFL event handler workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from p2pfl.learning.frameworks.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.management.logger import logger
from p2pfl.stages.network_state.basic_network_state import BasicNetworkState
from p2pfl.stages.workflows.models.event_handler_model import EventHandlerWorkflowModel

if TYPE_CHECKING:
    from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
    from p2pfl.node import Node

def get_states() -> list[dict]:
    """Define the states for the workflow."""
    states = [
        {'name': 'waitingContextUpdate'},
        {'name': "trainingFinished", 'final': True},
    ]
    return states

def get_transitions() -> list[dict]:
    """Define the transitions for the workflow."""
    transitions = [
        {'trigger': 'node_started', 'source': 'waitingContextUpdate', 'dest': None,
        'prepare': 'create_peer', 'conditions': 'is_all_nodes_started', 'after': 'send_network_ready'},

        {'trigger': 'peer_round_updated', 'source': 'waitingContextUpdate', 'dest': None,
        'prepare': 'save_peer_round_updated', 'conditions': 'is_all_models_initialized', 'after': 'send_peers_ready'},
        {'trigger': 'full_model_received', 'source': 'waitingContextUpdate', 'dest': None,
        'prepare': 'save_full_model', 'after': 'send_full_model_ready'},

        {'trigger': 'vote', 'source': 'waitingContextUpdate', 'dest': None,
        'prepare': 'save_votes', 'conditions': 'is_all_votes_received', 'after': 'send_votes_ready'},

        {'trigger': 'aggregated_models_received', 'source': 'waitingContextUpdate', 'dest': None,
        'prepare': 'save_aggregated_models'},
        {'trigger': 'aggregate', 'source': 'waitingContextUpdate', 'dest': None,
        'prepare': 'save_aggregation', 'conditions': 'is_all_models_received', 'after': 'send_aggregation_ready'},

        {'trigger': 'training_finished', 'source': '*', 'dest': 'trainingFinished'},
    ]
    return transitions

class BasicEventHandlerWorkflowModel(EventHandlerWorkflowModel):
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
    def node_started(self, source: str) -> bool:
        """Handle the node started event."""
        raise RuntimeError("Should be overridden!")

    def peer_round_updated(self, source: str, round: int) -> bool:
        """Handle the peer round updated event."""
        raise RuntimeError("Should be overridden!")

    def full_model_received(self, source: str, round: int, weights: bytes) -> bool:
        """Handle the full model received event."""
        raise RuntimeError("Should be overridden!")

    def vote(self, source: str, round: int, tmp_votes: list[tuple[str, int]]) -> bool:
        """Handle the vote event."""
        raise RuntimeError("Should be overridden!")

    def aggregated_models_received(self, source: str, round: int, aggregated_models: list[str]) -> bool:
        """Handle the aggregated models received event."""
        raise RuntimeError("Should be overridden!")

    def aggregate(self, model: P2PFLModel, source: str) -> bool:
        """Handle the aggregate event."""
        raise RuntimeError("Should be overridden!")


    ########################
    # EVENT HANDLER EVENTS #
    ########################
    async def send_network_ready(self, *args, **kwargs):
        """Send the network ready event."""
        await self.node.get_node_workflow().get_learning_workflow().network_ready()
        logger.info(self.node.address, "✅ Network ready.")

    async def send_models_ready(self, *args, **kwargs):
        """Send the models updated event."""
        await self.node.get_node_workflow().get_learning_workflow().models_ready()
        logger.info(self.node.address, "✅ Models ready.")

    async def send_votes_ready(self, *args, **kwargs):
        """Send the votes ready event."""
        await self.node.get_node_workflow().get_learning_workflow().votes_ready()
        logger.info(self.node.address, "✅ Votes ready.")

    async def send_aggregation_ready(self, *args, **kwargs):
        """Send the aggregation ready event."""
        await self.node.get_node_workflow().get_learning_workflow().aggregation_ready()
        logger.info(self.node.address, "✅ Aggregation ready.")

    async def send_full_model_ready(self, *args, **kwargs):
        """Send the full model ready event."""
        await self.node.get_node_workflow().get_learning_workflow().full_model_ready()
        logger.info(self.node.address, "✅ Full model ready.")

    async def send_peers_ready(self, *args, **kwargs):
        """Send the peers ready event."""
        await self.node.get_node_workflow().get_learning_workflow().peers_ready()
        logger.info(self.node.address, "✅ Peers ready.")

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
        self.network_state.update_round(source, -1)
        logger.debug(self.node.address, f"📦 Node started: {source}")

    async def save_votes(self, source: str, round: int, tmp_votes: list[tuple[str, int]]):
        """Save the votes."""
        local_state = self.node.get_local_state()
        if local_state is None:
            logger.error(self.node.address, f"Local state is None, cannot update votes for {source}")
            return

        if round == local_state.round:
            for train_set_id, vote in tmp_votes:
                self.network_state.add_vote(source, train_set_id, vote)
            logger.debug(self.node.address, f"📦 Votes received from {source}: {tmp_votes}")
        else:
            logger.error(self.node.address, f"📦 Votes not received from {source}: {tmp_votes} (expected {local_state.round})")

    async def save_aggregation(self, model: P2PFLModel, source: str):
        """Save the aggregation."""
        self.network_state.add_model(model, source)
        logger.debug(self.node.address, f"📦 Model received from {source}: {model}")

    async def create_peer(self, source: str):
        """Update the peer round."""
        self.network_state.add_peer(source)
        logger.debug(self.node.address, f"📡 {source} peer created")

    async def save_peer_round_updated(self,
                            source: str,
                            round: int,
                            ):
        """Initialize model."""
        local_round = self.node.get_local_state().round
        if local_round is None:
            logger.error(self.node.address, f"Local state is None, cannot update round for {source}")
            return

        if round in [local_round, local_round+1]:
            self.network_state.update_round(source, round)
            logger.debug(self.node.address, f"📡 Peer round updated: {source} -> {round}")
        else:
            logger.error(self.node.address, f"📡 Peer round not updated: {source} -> {round} (local round: {local_round})")


    async def save_full_model(self,
                            source: str,
                            round: int,
                            weights: bytes):
        """Initialize model."""
        # Check source
        logger.info(self.node.address, "📦 Full model received.")

        try:
            # Set new weights
            self.node.get_learner().set_P2PFLModel(weights)

            # Set model round
            self.node.get_learner().get_P2PFLModel().set_round(round)

            logger.info(self.node.address, "🤖 Model Weights Initialized")

        except DecodingParamsError:
            logger.error(self.node.address, "❌ Error decoding parameters.")
        except ModelNotMatchingError:
            logger.error(self.node.address, "❌ Models not matching.")
        except Exception as e:
            logger.error(self.node.address, f"❌ Unknown error adding model: {e}")

    async def save_aggregated_models(self, source: str, round: int, aggregated_models: list[str]):
        """Save the aggregated models."""
        local_state = self.node.get_local_state()
        if round == local_state.round:
            for aggregated_model in list(aggregated_models):
                self.network_state.add_aggregated_from(source, aggregated_model)

            logger.debug(self.node.address, f"📦 Aggregated models received from {source}: {aggregated_models}")
        else:
            logger.error(self.node.address, f"📦 Aggregated models not received from {source}: {aggregated_models} (expected {local_state.round})")



    ##############
    # CONDITIONS #
    ##############
    def is_all_nodes_started(self, *args, **kwargs):
        """Check if all nodes have started."""
        return len(self.network_state.list_peers()) == (len(self.node.communication_protocol.get_neighbors(only_direct=False)) + 1)

    def is_all_models_initialized(self, *args, **kwargs):
        """Check if all models have been initialized."""
        rounds = self.network_state.get_all_rounds()
        current_round = self.node.get_local_state().round
        initialized_nodes = sum(1 for value in rounds.values() if value == current_round)

        return initialized_nodes >= len(rounds)

    def is_all_votes_received(self, *args, **kwargs):
        """Check if all votes from neis have been received."""
        neighbors = self.network_state.list_peers()
        votes = self.network_state.get_all_votes()

        # Check if all neighbors have voted
        return all(votes.get(nei) for nei in neighbors)

    def is_all_models_received(self, *args, **kwargs):
        """Check if all models have been received."""
        return len(self.node.get_local_state().train_set) == len(self.network_state.get_all_models())
