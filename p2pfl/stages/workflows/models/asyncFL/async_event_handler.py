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

from p2pfl.learning.frameworks.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.management.logger import logger
from p2pfl.stages.network_state.async_network_state import AsyncNetworkState
from p2pfl.stages.workflows.models.event_handler_model import EventHandlerWorkflowModel

if TYPE_CHECKING:
    from p2pfl.node import Node


# Define states and transitions
def get_states() -> list[dict]:
    """Define the states for the workflow."""
    states = [
        dict(name='waitingContextUpdate', on_enter='on_enter_waiting_context_update'),
        dict(name='trainingFinished', final=True),
    ]

    return states

def get_transitions() -> list[dict]:
    """Define the transitions for the workflow."""
    transitions = [
        {'trigger': 'node_started', 'source': 'waitingContextUpdate', 'dest': None,
        'prepare': 'create_peer', 'conditions': 'is_all_nodes_started', 'after': 'send_network_ready'},

        {'trigger': 'loss_information_received', 'source': 'waitingContextUpdate', 'dest': None,
        'prepare': 'save_loss_information'},
        {'trigger': 'iteration_index_received', 'source': 'waitingContextUpdate', 'dest': None,
        'prepare': 'save_iteration_index'},
        {'trigger': 'model_received', 'source': 'waitingContextUpdate', 'dest': None,
        'prepare': 'save_model'},
        {'trigger': 'push_sum_weight_received', 'source': 'waitingContextUpdate', 'dest': None,
        'prepare': 'save_push_sum_weight'},

        {'trigger': 'training_finished', 'source': '*', 'dest': 'trainingFinished'},
    ]

    return transitions

class AsyncEventHandlerWorkflowModel(EventHandlerWorkflowModel):
    """Model for the training workflow."""

    def __init__(self, node: Node, network_state: AsyncNetworkState) -> None:
        """Initialize the workflow model."""
        self.network_state: AsyncNetworkState = network_state

        super().__init__(
            node=node,
        )

    ########################################
    # EVENTS (Overridden by pytransitions) #
    ########################################
    def node_started(self, source: str) -> bool:
        """Handle the node started event."""
        raise RuntimeError("Should be overridden!")

    def loss_information_received(self, source: str, round: int, loss: float) -> bool:
        """Handle the loss information received event."""
        raise RuntimeError("Should be overridden!")

    def iteration_index_received(self, source: str, index: int) -> bool:
        """Handle the iteration index received event."""
        raise RuntimeError("Should be overridden!")

    def model_received(self, source: str, round: int, weights: bytes, num_samples: int, contributors: list[str]) -> bool:
        """Handle the model received event."""
        raise RuntimeError("Should be overridden!")

    def push_sum_weight_received(self, source: str, push_sum_weight: float) -> bool:
        """Handle the push sum weight received event."""
        raise RuntimeError("Should be overridden!")

    def training_finished(self) -> bool:
        """Handle the training finished event."""
        raise RuntimeError("Should be overridden!")


    #########################
    # EVENT HANDLER SETTERS #
    #########################
    async def create_peer(self, source: str):
        """Update the peer round."""
        try:
            self.network_state.add_peer(source)
            logger.debug(self.node.address, f"📡 {source} peer created")
        except Exception as e:
            logger.error(self.node.address, f"❌ Error creating peer {source}: {e}")

    async def save_started_node(self, source: str):
        """Save the votes."""
        try:
            self.network_state.update_round(source, -1)
            logger.debug(self.node.address, f"📦 Node started: {source}")
        except Exception as e:
            logger.error(self.node.address, f"❌ Error saving started node {source}: {e}")

    async def save_loss_information(self, source: str, round: int, loss: float):
        """Save the loss information."""
        try:
            self.network_state.add_loss(source, round, loss)
            logger.debug(self.node.address, f"📡 {source} loss updated to {loss} for round {round}")
        except Exception as e:
            logger.error(self.node.address, f"❌ Error saving loss information from {source} for round {round}: {e}")

    async def save_model(self,
        source: str,
        round: int,
        weights: bytes,
        num_samples: int,
        contributors: list[str]
        ):
        """Initialize model."""
        # Check source
        # Wait and gossip model initialization
        logger.info(self.node.address, "📦 Model received.")

        try:
            model = self.node.get_learner().get_P2PFLModel().build_copy(params=weights, num_samples=num_samples, contributors=contributors)
            self.network_state.add_model(model, source)

        except DecodingParamsError:
            logger.error(self.node.address, "❌ Error decoding parameters.")
        except ModelNotMatchingError:
            logger.error(self.node.address, "❌ Models not matching.")
        except Exception as e:
            logger.error(self.node.address, f"❌ Unknown error adding model: {e}")

    async def save_iteration_index(self,
                            source: str,
                            index: int,
                            ):
        """Save the iteration index."""
        try:
            self.network_state.update_round(source, index)
            logger.debug(self.node.address, f"📡 {source} round updated to {index}")
        except Exception as e:
            logger.error(self.node.address, f"❌ Error saving iteration index from {source}: {e}")

    async def save_push_sum_weight(self, source: str, push_sum_weight: float):
        """Save the push sum weight."""
        try:
            self.network_state.update_push_sum_weight(source, push_sum_weight)
            logger.debug(self.node.address, f"📡 {source} push sum weight updated to {push_sum_weight}")
        except Exception as e:
            logger.error(self.node.address, f"❌ Error saving push sum weight from {source}: {e}")

    ##############
    # CONDITIONS #
    ##############
    def is_all_nodes_started(self, *args, **kwargs):
        """Check if all nodes have started."""
        return len(self.network_state.list_peers()) == (len(self.node.communication_protocol.get_neighbors(only_direct=True)) + 1)

    ########################
    # EVENT HANDLER EVENTS #
    ########################
    async def send_network_ready(self, *args, **kwargs):
        """Send the network ready event."""
        logger.info(self.node.address, "✅ Network ready.")
        await self.node.get_node_workflow().get_learning_workflow().network_ready()
