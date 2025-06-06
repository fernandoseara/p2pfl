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

import asyncio
import collections
from typing import TYPE_CHECKING

from transitions.extensions.nesting import NestedState

from p2pfl.management.logger import logger

NestedState.separator = '↦'

if TYPE_CHECKING:
    from p2pfl.node import Node


class LearningWorkflowModel:
    """Model for the training workflow."""

    def __init__(self, node: Node, state_history_length: int = 10):
        """Initialize the workflow model."""
        self.node = node
        self.candidates: list = []

        #self.event_log: list = []
        self.state_log: collections.deque = collections.deque(maxlen=state_history_length)

    @property
    def state(self):
        """Get the current state of the workflow."""
        return self.state_log[-1]

    @state.setter
    def state(self, value):
        """Set the current state of the workflow."""
        self.state_log.append(value)

    @property
    def waiting_for_learning_start(self) -> bool:
        """
        Check if the workflow is waiting for the learning start.

        Returns:
            bool: True if the workflow is waiting for the learning start, False otherwise.

        """
        return self.is_waiting_for_learning_start()

    @property
    def finished(self) -> bool:
        """
        Check if the workflow is finished.

        Returns:
            bool: True if the workflow is finished, False otherwise.

        """
        return self.is_learning_finished()

    async def set_model_initialized(self, *args, **kwargs):
        """Set the model initialized."""
        # Set the model initialized
        self.node.get_learner().get_P2PFLModel().set_round(0)


    ###################
    # STATE CALLBACKS #
    ###################
    async def on_final_learning(self):
        """Finish the learning workflow."""
        await self.learning_finished()


    ######################
    # LOGGING CALLBACKS #
    ######################
    def finalize_logging(self, *args, **kwargs):
        """Logging callback."""
        logger.debug(self.node.address, f"🏃 Running stage: {(self.state)}")

    def test(self, *args, **kwargs):
        """Test function for debugging."""
        logger.info(self.node.address, "Test function called.")


    #############
    # INTERRUPT #
    #############
    async def interrupt(self):
        """Interrupt the workflow."""
        global machine
        await asyncio.sleep(1)
        for task in machine.async_tasks[id(self)]:
            task.cancel()
        machine._transition_queue_dict[id(self)].clear()

        await self.stop()
