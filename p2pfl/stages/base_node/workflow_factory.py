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

from p2pfl.stages.workflow_factory import WorkflowFactory

if TYPE_CHECKING:
    from p2pfl.communication.commands.command import Command
    from p2pfl.node import Node
    from p2pfl.stages.workflows import TrainingWorkflow


class BasicDFLFactory(WorkflowFactory):
    """Factory class to create workflows. Main goal: Avoid cyclic imports."""

    @staticmethod
    def create_training_workflow() -> type[TrainingWorkflow]:
        """Create a workflow."""
        from p2pfl.stages.base_node.workflow import BasicDFLWorkflow

        return BasicDFLWorkflow

    @staticmethod
    def create_commands(node: Node) -> list[Command]:
        """Create a list of commands."""
        from p2pfl.communication.commands.message.metrics_command import MetricsCommand
        from p2pfl.communication.commands.message.models_agregated_command import ModelsAggregatedCommand
        from p2pfl.communication.commands.message.node_initialized_command import NodeInitializedCommand
        from p2pfl.communication.commands.message.peer_round_updated_command import PeerRoundUpdatedCommand
        from p2pfl.communication.commands.message.start_learning_command import StartLearningCommand
        from p2pfl.communication.commands.message.stop_learning_command import StopLearningCommand
        from p2pfl.communication.commands.message.vote_train_set_command import VoteTrainSetCommand
        from p2pfl.communication.commands.weights.full_model_command import FullModelCommand
        from p2pfl.communication.commands.weights.partial_model_command import PartialModelCommand

        return [
            StartLearningCommand(node),
            StopLearningCommand(node),
            NodeInitializedCommand(node),
            PeerRoundUpdatedCommand(node),
            VoteTrainSetCommand(node),
            ModelsAggregatedCommand(node),
            MetricsCommand(node),
            PartialModelCommand(node),
            FullModelCommand(node),
        ]
