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

"""Stage factory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from p2pfl.stages.workflow_type import WorkflowType

if TYPE_CHECKING:  # Only imports the below statements during type checking
    from p2pfl.node import Node
    from p2pfl.stages.workflows.workflow_state_manager import WorkflowStateManager


class WorkflowBuilder(ABC):
    """Command interface."""

    @property
    @abstractmethod
    def workflow_state_manager(self) -> WorkflowStateManager:
        """Get the resulting workflow."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the builder to initial state."""
        raise NotImplementedError

    @abstractmethod
    def create_states(self) -> None:
        """Create states."""
        raise NotImplementedError

    @abstractmethod
    def create_transitions(self) -> None:
        """Create transitions."""
        raise NotImplementedError

    @abstractmethod
    def create_network_state(self) -> None:
        """Create network state."""
        raise NotImplementedError

    @abstractmethod
    def create_workflow_machine(self) -> None:
        """Create a workflow."""
        raise NotImplementedError

    @abstractmethod
    def create_event_handler_machine(self) -> None:
        """Create event handler machine."""
        raise NotImplementedError

    @abstractmethod
    def create_training_workflow_model(self, node: Node) -> None:
        """Create a workflow model."""
        raise NotImplementedError

    @abstractmethod
    def create_event_handler_workflow_model(self, node: Node) -> None:
        """Create event handler workflow model."""
        raise NotImplementedError

    @abstractmethod
    def create_commands(self, node: Node) -> None:
        """Create commands list."""
        raise NotImplementedError

    def create_model(self, node: Node) -> None:
        """Create model."""
        pass



# Abstract factory pattern
class WorkflowBuilderFactory:
    """Factory producer for workflow builders."""

    @staticmethod
    def get_builder(workflow_name: str) -> type[WorkflowBuilder]:
        """Return the stage class."""
        if workflow_name == WorkflowType.BASIC.value:
            from p2pfl.stages.workflows.builder.basicDFL.workflow_builder import BasicDFLWorkflowBuilder

            return BasicDFLWorkflowBuilder
        elif workflow_name == WorkflowType.ASYNC.value:
            from p2pfl.stages.workflows.builder.asyncFL.asydfl_workflow_builder import AsyDFLWorkflowBuilder
            return AsyDFLWorkflowBuilder
        else:
            raise Exception("Invalid workflow name.")
