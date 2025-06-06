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
from transitions.extensions.nesting import NestedState

NestedState.separator = '↦'


class AsyncEventHandlerWorkflow(BaseMachine):
    """
    Event handler model for the base node.

    This class is used to handle the events of the base node.
    It uses the transitions library to create a state machine.

    """

    def __init__(self, model=None):
        """Initialize the event handler model."""
        states = [
            {'name': 'waiting_context_update', 'on_enter': 'on_enter_waiting_context_update'},
            {'name': "training_finished", 'final': True},
        ]

        transitions = [
            {'trigger': 'node_started', 'source': 'waiting_context_update', 'dest': None,
            'prepare': 'create_peer', 'conditions': 'is_all_nodes_started', 'after': 'send_network_ready'},

            {'trigger': 'loss_information_received', 'source': 'waiting_context_update', 'dest': None,
            'prepare': 'save_loss_information'},
            {'trigger': 'iteration_index_received', 'source': 'waiting_context_update', 'dest': None,
            'prepare': 'save_iteration_index'},
            {'trigger': 'model_received', 'source': 'waiting_context_update', 'dest': None,
            'prepare': 'save_model'},
            {'trigger': 'push_sum_weight_received', 'source': 'waiting_context_update', 'dest': None,
            'prepare': 'save_push_sum_weight'},

            {'trigger': 'training_finished', 'source': '*', 'dest': 'training_finished'},
        ]

        super().__init__(model=model,
                         states=states,
                         transitions=transitions,
                         initial='waiting_context_update',
                         queued=True,
                         ignore_invalid_triggers=True
        )
