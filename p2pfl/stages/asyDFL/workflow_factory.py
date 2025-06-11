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

from typing import TYPE_CHECKING

from p2pfl.communication.commands.command import Command
from p2pfl.communication.commands.message.node_initialized_command import NodeInitializedCommand
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.node import Node
from p2pfl.stages.workflow_factory import WorkflowFactory
from p2pfl.stages.workflows.workflows import LearningWorkflow

if TYPE_CHECKING:
    from p2pfl.communication.commands.command import Command
    from p2pfl.node import Node
    from p2pfl.stages.workflows.workflows import LearningWorkflow

class AsyDFLFactory(WorkflowFactory):
    """Factory class to create workflows."""

    @staticmethod
    def create_training_workflow(node: Node) -> LearningWorkflow:
        """Create a workflow."""
        from p2pfl.stages.workflows.async_event_handler_workflow import AsyncEventHandlerWorkflow
        from p2pfl.stages.workflows.async_training_workflow import AsyncTrainingWorkflow
        from p2pfl.stages.workflows.models.async_learning_workflow_model import AsyncLearningWorkflowModel

        model = AsyncLearningWorkflowModel(node)
        return model, LearningWorkflow(model, AsyncTrainingWorkflow(), AsyncEventHandlerWorkflow())

    @staticmethod
    def create_commands(node: Node) -> list[Command]:
        """Create a workflow."""
        from p2pfl.communication.commands.message.asyDFL import (
            IndexInformationUpdatingCommand,
            LossInformationUpdatingCommand,
            ModelInformationUpdatingCommand,
            PushSumWeightInformationUpdatingCommand,
        )
        from p2pfl.communication.commands.message.peer_round_updated_command import PeerRoundUpdatedCommand

        return [
            NodeInitializedCommand(node),
            PeerRoundUpdatedCommand(node),
            LossInformationUpdatingCommand(node),
            IndexInformationUpdatingCommand(node),
            ModelInformationUpdatingCommand(node),
            PushSumWeightInformationUpdatingCommand(node),
        ]

    @staticmethod
    def create_model(model: P2PFLModel) -> P2PFLModel:
        """Create model."""
        from p2pfl.learning.frameworks.custom_model_factory import CustomModelFactory

        return CustomModelFactory.create_model("AsyDFL", model)
