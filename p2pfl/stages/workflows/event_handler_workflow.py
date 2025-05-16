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

from __future__ import annotations

from transitions.extensions import HierarchicalAsyncGraphMachine as BaseMachine
from transitions.extensions.nesting import NestedState

NestedState.separator = '↦'


class EventHandlerWorkflow(BaseMachine):
    """
    Event handler model for the base node.

    This class is used to handle the events of the base node.
    It uses the transitions library to create a state machine.

    """

    def __init__(self, model=None):
        """Initialize the event handler model."""
        states = [
            {'name': "waiting_network_start", 'on_enter': 'on_enter_waiting_network_start'},
            {'name': 'waiting_model_update', 'on_enter': 'on_enter_waiting_network_start'},
            {'name': "waiting_vote", 'on_enter': 'on_enter_waiting_vote'},
            {'name': "votes_received"},
            {'name': "waiting_partial_model", 'on_enter': 'on_enter_waiting_partial_model'},
            {'name': "training_finished", 'final': True},
        ]

        transitions = [
            {'trigger': 'node_started', 'source': 'waiting_network_start', 'dest': 'waiting_model_update',
            'prepare': 'create_peer', 'conditions': 'is_all_nodes_started', 'after': 'send_network_ready'},

            {'trigger': 'peer_round_updated', 'source': 'waiting_model_update', 'dest': 'waiting_vote',
            'prepare': 'save_peer_round_updated', 'conditions': 'is_all_models_initialized', 'after': 'send_peers_ready'},
            {'trigger': 'full_model_received', 'source': 'waiting_model_update', 'dest': None,
            'prepare': 'save_full_model', 'after': 'send_full_model_ready'},

            {'trigger': 'vote', 'source': 'waiting_vote', 'dest': 'votes_received',
            'prepare': 'save_votes', 'conditions': 'is_all_votes_received', 'after': 'send_votes_ready'},

            {'trigger': 'votes_ready', 'source': 'votes_received', 'dest': 'waiting_partial_model', 'conditions': 'in_train_set'},
            {'trigger': 'votes_ready', 'source': 'votes_received', 'dest': 'waiting_model_update'},

            {'trigger': 'aggregate', 'source': 'waiting_partial_model', 'dest': 'waiting_model_update',
            'prepare': 'save_aggregation', 'conditions': 'is_all_models_received', 'after': 'send_aggregation_ready'},

            {'trigger': 'training_finished', 'source': '*', 'dest': 'training_finished'},
        ]

        super().__init__(model=model,
                         states=states,
                         transitions=transitions,
                         initial='waiting_network_start',
                         queued=True,
                         ignore_invalid_triggers=True
        )
