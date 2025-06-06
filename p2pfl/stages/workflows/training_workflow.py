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

from typing import TYPE_CHECKING

from transitions.extensions import HierarchicalAsyncMachine as BaseMachine
from transitions.extensions.asyncio import AsyncTimeout
from transitions.extensions.nesting import NestedState
from transitions.extensions.states import add_state_features

from p2pfl.settings import Settings

if TYPE_CHECKING:
    from p2pfl.node import Node

NestedState.separator = '↦'

@add_state_features(AsyncTimeout)
class TimeoutMachine(BaseMachine):
    """State machine with timeout support."""

    pass


class BasicTrainingWorkflow(TimeoutMachine):
    """
    Event handler model for the base node.

    This class is used to handle the events of the base node.
    It uses the transitions library to create a state machine.

    """

    def __init__(self, model=None):
        """Initialize the event handler model."""
        states = [
            # Setup & initial synchronization
            {'name': "starting_training", 'on_enter': 'on_enter_starting_training'},
            {'name': "waiting_for_synchronization", 'on_enter': 'on_enter_waiting_for_synchronization'},
            {'name': "nodes_synchronized", 'on_enter': 'on_enter_nodes_synchronized'},
            {'name': "waiting_for_full_model"},
            {'name': "updating_round", 'on_enter': 'on_enter_updating_round'},
            {'name': "gossiping_full_model", 'on_enter': 'on_enter_gossipping_full_model'},
            {'name': "waiting_for_network_start", 'on_timeout': "gossip_timeout"},
            {'name': "round_initialized", 'on_enter': 'on_enter_round_initialized'},
            {'name': 'p2p_voting', 'initial': 'starting_voting', 'on_final': 'on_final_p2p_voting', 'children': [
                {'name': "starting_voting", 'on_enter': 'on_enter_starting_voting'},
                {'name': "voting", 'on_enter': 'on_enter_voting'},
                {'name': "waiting_voting", 'timeout': Settings.training.VOTE_TIMEOUT, 'on_timeout': "voting_timeout"},
                {'name': 'voting_finished', 'on_enter': 'on_enter_voting_finished', 'final': True},
            ]},
            {'name': 'p2p_learning', 'initial': 'evaluating', 'on_final': 'on_final_p2p_learning', 'children': [
                {'name': 'evaluating', 'on_enter': 'on_enter_evaluating'},
                {'name': 'training', 'on_enter': 'on_enter_training'},
                {'name': 'gossipping_partial_aggregation', 'on_enter': 'on_enter_gossipping_partial_aggregation'},
                {'name': "waiting_for_partial_aggregation", 'timeout': Settings.training.AGGREGATION_TIMEOUT, 'on_timeout': "aggregation_timeout"},
                {'name': 'aggregating', 'on_enter': 'on_enter_aggregating'},
                {'name': "aggregation_finished", 'on_enter': 'on_enter_aggregation_finished', 'final': True},
            ]},
            {'name': "round_finished", 'on_enter': 'on_enter_round_finished'},
            {'name': "training_finished", 'on_enter': 'on_enter_training_finished', 'final': True},
        ]

        transitions = [
            # Setup & initial synchronization
            {'trigger': 'next_stage', 'source': 'starting_training', 'dest': 'waiting_for_synchronization'},
            {'trigger': 'network_ready', 'source': 'waiting_for_synchronization', 'dest': 'nodes_synchronized'},

            # Model initialization
            {'trigger': 'next_stage', 'source': 'nodes_synchronized', 'dest': 'updating_round', 'conditions': 'is_model_initialized'},
            {'trigger': 'next_stage', 'source': 'nodes_synchronized', 'dest': 'waiting_for_full_model'},
            {'trigger': 'full_model_ready', 'source': 'waiting_for_full_model', 'dest': 'updating_round'},

            # Update round
            {'trigger': 'continue_p2p_round_initialization', 'source': 'updating_round', 'dest': 'gossiping_full_model',
            'prepare': ['get_full_gossipping_candidates'], 'conditions': 'candidate_exists'},
            {'trigger': 'continue_p2p_round_initialization', 'source': 'updating_round', 'dest': 'waiting_for_network_start'},

            # Gossip full model
            {'trigger': 'continue_p2p_round_initialization', 'source': 'gossiping_full_model', 'dest': 'round_initialized',
            'conditions': 'is_all_models_initialized'},
            {'trigger': 'continue_p2p_round_initialization', 'source': 'gossiping_full_model', 'dest': 'waiting_for_network_start'},
            {'trigger': 'peers_ready', 'source': 'waiting_for_network_start', 'dest': 'round_initialized'},

            # Workflow finish check
            {'trigger': 'next_stage', 'source': 'round_initialized', 'dest': 'training_finished', 'conditions': 'is_total_rounds_reached'},
            {'trigger': 'next_stage', 'source': 'round_initialized', 'dest': 'p2p_voting'},

            # Voting
            {'trigger': 'continue_p2p_voting', 'source': 'p2p_voting↦starting_voting', 'dest': 'p2p_voting↦voting'},
            {'trigger': 'continue_p2p_voting', 'source': 'p2p_voting↦voting', 'dest': 'p2p_voting↦waiting_voting'},
            {'trigger': 'votes_ready', 'source': 'p2p_voting', 'dest': 'p2p_voting↦voting_finished'},
            {'trigger': 'voting_timeout', 'source': 'p2p_voting↦waiting_voting', 'dest': 'p2p_voting↦voting_finished'},

            # Voting outcome
            {'trigger': 'next_stage', 'source': 'p2p_voting', 'dest': 'p2p_learning', 'conditions': 'in_train_set'},
            {'trigger': 'next_stage', 'source': 'p2p_voting', 'dest': 'waiting_for_full_model'},

            # Learning
            {'trigger': 'continue_p2p_learning', 'source': 'p2p_learning↦evaluating', 'dest': 'p2p_learning↦training'},
            {'trigger': 'continue_p2p_learning', 'source': 'p2p_learning↦training', 'dest': 'p2p_learning↦gossipping_partial_aggregation',
            'prepare': ['get_partial_gossipping_candidates'], 'conditions': 'candidate_exists'},
            {'trigger': 'continue_p2p_learning', 'source': 'p2p_learning↦training', 'dest': 'p2p_learning↦waiting_for_partial_aggregation'},
            {'trigger': 'continue_p2p_learning', 'source': 'p2p_learning↦gossipping_partial_aggregation',
             'dest': 'p2p_learning↦waiting_for_partial_aggregation'},

            {'trigger': 'aggregation_ready', 'source': 'p2p_learning', 'dest': 'p2p_learning↦aggregating'},
            {'trigger': 'aggregation_timeout', 'source': 'p2p_learning↦waiting_for_partial_aggregation',
             'dest': 'p2p_learning↦aggregating'},

            {'trigger': 'continue_p2p_learning', 'source': 'p2p_learning↦aggregating', 'dest': 'p2p_learning↦aggregation_finished'},

            # Loop
            {'trigger': 'next_stage', 'source': 'p2p_learning', 'dest': 'round_finished'},
            {'trigger': 'next_stage', 'source': 'round_finished', 'dest': 'updating_round', 'conditions': 'is_all_models_received'},
            {'trigger': 'next_stage', 'source': 'round_finished', 'dest': 'waiting_for_full_model'},
        ]

        super().__init__(model=model,
                         states=states,
                         transitions=transitions,
                         initial='starting_training',
                         queued='model',
                         ignore_invalid_triggers=True,
        )
