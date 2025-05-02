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

from transitions.extensions import HierarchicalAsyncMachine

from p2pfl.management.logger import logger

states = [
    {'name': "waiting_network_start", 'on_enter': 'on_enter_waiting_network_start'},
    {'name': 'waiting_model_update', 'parallel': [
        {'name': 'waiting_round', 'initial': 'round', 'children': [
            {'name': "round", 'on_enter': 'on_enter_waiting_round_update'},
            {'name': "rounds_updated", 'final': True},
        ]},
        {'name': 'waiting_full_model', 'initial': 'full_model', 'children': [
            {'name': "full_model", 'on_enter': 'on_enter_waiting_full_model'},
            {'name': "full_models_updated", 'final': True},
        ]},
    ], 'on_final': "send_models_ready"},
    {'name': "waiting_vote", 'on_enter': 'on_enter_waiting_vote'},
    {'name': "waiting_partial_model", 'on_enter': 'on_enter_waiting_partial_model'},
]

transitions = [
    {
        'trigger': 'node_started',
        'source': 'waiting_network_start',
        'dest': 'waiting_model_update',
        'prepare': 'create_peer',
        'conditions': 'is_all_nodes_started',
        'after': 'send_network_ready'
    },
    {
        'trigger': 'peer_round_updated',
        'source': 'waiting_model_update↦round',
        'dest': 'waiting_model_update↦rounds_updated',
        'prepare': 'save_peer_round_updated',
        'conditions': 'is_all_models_initialized',
        'after': 'send_peers_ready'
    },
    {
        'trigger': 'full_model_received',
        'source': 'waiting_model_update↦full_model',
        'dest': 'waiting_model_update↦full_models_updated',
        'prepare': 'save_full_model',
        'after': 'send_full_model_ready'
    },
    {
        'trigger': 'models_ready',
        'source': 'waiting_model_update',
        'dest': 'waiting_vote',
        'conditions': 'in_train_set'
    },
    {
        'trigger': 'models_ready',
        'source': 'waiting_model_update',
        'dest': None
    },
    {
        'trigger': 'vote',
        'source': 'waiting_vote',
        'dest': 'waiting_partial_model',
        'prepare': 'save_votes',
        'conditions': 'is_all_votes_received',
        'after': 'send_votes_ready'
    },
    {
        'trigger': 'aggregate',
        'source': 'waiting_partial_model',
        'dest': 'waiting_model_update',
        'prepare': 'save_aggregation',
        'conditions': 'is_all_models_received',
        'after': 'send_aggregation_ready'
    }
]


class EventHandlerModel(object):
    """
    Event handler model for the base node.

    This class is used to handle the events of the base node.
    It uses the transitions library to create a state machine.

    """
    ###########################
    # EVENT HANDLER CALLBACKS #
    ###########################
    async def on_enter_waiting_network_start(self, *args, **kwargs):
        """Wait for the node to start."""
        logger.info(self.node.address, "⏳ Waiting for the node to start.")

    async def on_enter_waiting_full_model(self, *args, **kwargs):
        """Wait for the full model."""
        logger.info(self.node.address, "⏳ Waiting for the full model.")

    async def on_enter_waiting_round_update(self, *args, **kwargs):
        """Wait for the round update."""
        logger.info(self.node.address, "⏳ Waiting for peer round updates.")

    async def on_enter_waiting_vote(self, *args, **kwargs):
        """Wait for the vote."""
        logger.info(self.node.address, "⏳ Waiting for votes.")

    async def on_enter_waiting_partial_model(self, *args, **kwargs):
        """Wait for the partial model."""
        logger.info(self.node.address, "⏳ Waiting for partial models.")

    ########################
    # EVENT HANDLER EVENTS #
    ########################
    async def send_network_ready(self, *args, **kwargs):
        """Send the network ready event."""
        await self.network_ready()
        logger.info(self.node.address, "✅ Network ready.")

    async def send_full_model_ready(self, *args, **kwargs):
        """Send the full model ready event."""
        await self.full_model_ready()
        logger.info(self.node.address, "✅ Full model ready.")

    async def send_peers_ready(self, *args, **kwargs):
        """Send the peers ready event."""
        await self.peers_ready()
        logger.info(self.node.address, "✅ Peers ready.")

    async def send_models_ready(self, *args, **kwargs):
        """Send the models updated event."""
        await self.models_ready()
        logger.info(self.node.address, "✅ Models ready.")

    async def send_votes_ready(self, *args, **kwargs):
        """Send the votes ready event."""
        await self.votes_ready()
        logger.info(self.node.address, "✅ Votes ready.")

    async def send_aggregation_ready(self, *args, **kwargs):
        """Send the aggregation ready event."""
        await self.aggregation_ready()
        logger.info(self.node.address, "✅ Aggregation ready.")

event_handler = HierarchicalAsyncMachine(states=states, transitions=transitions, initial='waiting_network_start')
