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

from transitions.extensions import HierarchicalAsyncMachine as BaseMachine
from transitions.extensions.asyncio import AsyncTimeout
from transitions.extensions.nesting import NestedState
from transitions.extensions.states import add_state_features

NestedState.separator = '↦'

@add_state_features(AsyncTimeout)
class TimeoutMachine(BaseMachine):
    """State machine with timeout support."""

    pass

from p2pfl.management.logger import logger

class AsyncTrainingWorkflow(TimeoutMachine):
    """
    Event handler model for the base node.

    This class is used to handle the events of the base node.
    It uses the transitions library to create a state machine.

    """

    def __init__(self, model=None):
        """Initialize the event handler model."""
        states = [
            {'name': "starting_training", 'on_enter': 'on_enter_starting_training'},
            {'name': "waiting_for_synchronization", 'on_enter': 'on_enter_waiting_for_synchronization'},
            {'name': "nodes_synchronized", 'on_enter': 'on_enter_nodes_synchronized'},
            {'name': 'training_round', 'initial': 'debiasing_model', 'on_final': 'on_final_training_round', 'children': [
                {'name': "debiasing_model", 'on_enter': 'on_enter_debiasing_model'},
                {'name': 'updating_local_model', 'on_enter': 'on_enter_updating_local_model'},
                {'name': "sending_training_loss", 'on_enter': 'on_enter_sending_training_loss'},

                {'name': 'network_updating', 'initial': 'gossiping_model', 'on_final': 'on_final_network_updating', 'children': [
                    {'name': "gossiping_model", 'on_enter': 'on_enter_gossipping_model'},
                    {'name': 'aggregating', 'on_enter': 'on_enter_aggregating'},
                    {'name': "finishing", 'on_enter': 'on_enter_network_updating_finishing', 'final': True},
                ]},

                {'name': "finishing", 'on_enter': 'on_enter_training_round_finishing', 'final': True},
            ]},
            {'name': "training_finished", 'on_enter': 'on_enter_training_finished', 'final': True},
        ]

        transitions = [
            # Setup & initial synchronization
            {'trigger': 'next_stage', 'source': 'starting_training', 'dest': 'waiting_for_synchronization'},
            {'trigger': 'network_ready', 'source': 'waiting_for_synchronization', 'dest': 'nodes_synchronized'},
            {'trigger': 'next_stage', 'source': 'nodes_synchronized', 'dest': 'training_round'},

            # Debiasing & updating
            {'trigger': 'next_stage', 'source': 'training_round↦debiasing_model', 'dest': 'training_round↦updating_local_model'},
            {'trigger': 'next_stage', 'source': 'training_round↦updating_local_model', 'dest': 'training_round↦sending_training_loss'},

            # Check if is it time to update the model with network updating
            {'trigger': 'next_stage', 'source': 'training_round↦sending_training_loss', 'dest': 'training_round↦network_updating',
             'conditions': 'is_iteration_network_updating', 'before': 'get_gossip_candidates'},
            {'trigger': 'next_stage', 'source': 'training_round↦sending_training_loss', 'dest': 'training_round↦finishing'},

            # Network updating
            {'trigger': 'next_stage', 'source': 'training_round↦network_updating↦gossiping_model', 'dest': 'training_round↦network_updating↦aggregating'},
            {'trigger': 'next_stage', 'source': 'training_round↦network_updating↦aggregating', 'dest': 'training_round↦network_updating↦finishing'},
            {'trigger': 'next_stage', 'source': 'training_round↦network_updating', 'dest': 'training_round↦finishing'},

            # Loop
            {'trigger': 'next_stage', 'source': 'training_round', 'dest': 'training_finished', 'conditions': 'is_total_rounds_reached'},
            {'trigger': 'next_stage', 'source': 'training_round', 'dest': 'nodes_synchronized'},
        ]

        super().__init__(
            model=model,
            states=states,
            transitions=transitions,
            initial='starting_training',
            queued='model',
            ignore_invalid_triggers=True,
        )
