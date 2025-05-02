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

from functools import partial
from typing import TYPE_CHECKING

from transitions.extensions import HierarchicalAsyncGraphMachine
from transitions.extensions.asyncio import AsyncTimeout
from transitions.extensions.nesting import NestedState
from transitions.extensions.states import add_state_features

from p2pfl.management.logger import logger

if TYPE_CHECKING:
    from p2pfl.node import Node




def _log_trigger(self, *args, **kwargs):
    """Log the trigger event."""
    logger.debug(self.node.address, f"🏃 Running stage: {(self.state)}")
    #self.event_log.append(self.event)
    self.state_log.append(self.state)


class TrainingWorkflow(object):
    """Class to run a workflow of stages."""

    @property
    def finished(self) -> bool:
        """
        Check if the workflow is finished.

        Returns:
            bool: True if the workflow is finished, False otherwise.

        """
        return self.is_training_finished()

    def __init__(self, node: Node, *args, **kwargs):
        """Initialize the workflow."""
        self.event_log: list[str] = []
        self.state_log: list[str] = []
        self.node = node

        training_workflow_states = 

        states = [
            {'name': "waiting_for_training_start"},
            {'name': "training", 'parallel': [
                {'name': 'workflow', 'initial': 'starting_training', 'children': training_workflow_states},
                {'name': 'event_handler', 'initial': 'waiting_network_start', 'children': event_handler_states}
            ]},
            {'name': "training_finished", 'on_enter': 'on_enter_training_finished', 'final': True}
        ]

        transitions = [
            {'trigger': 'start_learning', 'source': 'waiting_for_training_start', 'dest': 'training', 'after': 'set_model_initialized'},
            {'trigger': 'peer_learning_initiated', 'source': 'waiting_for_training_start', 'dest': 'training'},
        ]

        super().__init__(
            states=states,
            transitions=transitions,
            model=model,
            initial='waiting_for_training_start',
            queued='model',
            ignore_invalid_triggers=True,
            finalize_event=self._log_trigger,
            #on_final=partial(final_event_raised, 'TrainingWorkflow'),
            #ignore_invalid_triggers=True
        )

    def _log_trigger(self, *args, **kwargs):
        logger.debug(self.node.address, f"🏃 Running stage: {(self.state)}")
        #self.event_log.append(self.event)
        self.state_log.append(self.state)
