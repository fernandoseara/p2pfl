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

NestedState.separator = '↦'

@add_state_features(AsyncTimeout)
class TimeoutMachine(HierarchicalAsyncGraphMachine):
    """State machine with timeout support."""

    pass


class TrainingWorkflow(TimeoutMachine):
    """Class to run a workflow of stages."""

    @property
    def finished(self) -> bool:
        """Return the name of the workflow."""
        #return self.training_finished.is_active
        return False

    def __init__(self, node: Node, states=None, transitions=None, model=None, *args, **kwargs):
        """Initialize the workflow."""
        self.event_log: list[str] = []
        self.state_log: list[str] = []
        self.node = node
        self.is_running: bool = False

        states = [{'name': "waiting_for_training_start"},
            {'name': "training_finished", 'final':True}
        ] if states is None else states

        transitions = [
            {'trigger': 'start_training', 'source': 'waiting_for_training_start', 'dest': 'training_finished'},
        ] if transitions is None else transitions

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
