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
"""Director for building workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from p2pfl.node import Node
    from p2pfl.stages.workflows.builder.workflow_builder import WorkflowBuilder


class WorkflowDirector:
    """
    Director class to construct workflows using a builder.

    Attributes:
        builder: The builder instance.

    """

    def __init__(self) -> None:
        """Initialize the workflow director."""
        self._builder: WorkflowBuilder | None = None

    @property
    def builder(self) -> WorkflowBuilder | None:
        """
        Get the builder instance.

        Returns:
            The builder instance.

        """
        return self._builder

    @builder.setter
    def builder(self, builder: WorkflowBuilder) -> None:
        """
        Set a builder instance to construct a workflow.

        Args:
            builder: The builder instance.

        """
        self._builder = builder

    def build_workflow_state_manager(self, node: Node) -> None:
        """
        Construct a workflow state manager using the builder interface.

        Args:
            node: The node instance.

        Returns:
            The constructed workflow state manager.

        """
        if self._builder is None:
            raise ValueError("Builder is not set.")

        self._builder.reset()
        self._builder.create_commands(node)
        self._builder.create_model(node)
        self._builder.create_states()
        self._builder.create_transitions()
        self._builder.create_network_state()
        self._builder.create_workflow_machine()
        self._builder.create_event_handler_machine()
        self._builder.create_training_workflow_model(node)
        self._builder.create_event_handler_workflow_model(node)
