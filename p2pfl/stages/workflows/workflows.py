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

from transitions.extensions import HierarchicalAsyncGraphMachine
from transitions.extensions.nesting import NestedState

from p2pfl.management.logger import logger
from p2pfl.stages.workflows.event_handler_workflow import EventHandlerWorkflow
from p2pfl.stages.workflows.training_workflow import TrainingWorkflow

NestedState.separator = '↦'


class LearningWorkflow(HierarchicalAsyncGraphMachine):
    """Class to run a workflow of stages."""

    def __init__(self, model, *args, **kwargs):
        """Initialize the workflow."""
        self.event_log: list[str] = []
        self.state_log: list[str] = []

        states = [
            {'name': "waiting_for_training_start"},
            {'name': "training", 'parallel': [{"name":"workflow","children":TrainingWorkflow()}, {"name":"event_handler","children":EventHandlerWorkflow()}]},
            {'name': "training_finished", 'final': True}
        ]

        transitions = [
            {'trigger': 'start_learning', 'source': 'waiting_for_training_start', 'dest': 'training', 'after': 'set_model_initialized'},
            {'trigger': 'peer_learning_initiated', 'source': 'waiting_for_training_start', 'dest': 'training'},
        ]

        super().__init__(
            model=model,
            states=states,
            transitions=transitions,
            initial='waiting_for_training_start',
            queued='model',
            ignore_invalid_triggers=True,
            finalize_event='finalize_logging',
        )
