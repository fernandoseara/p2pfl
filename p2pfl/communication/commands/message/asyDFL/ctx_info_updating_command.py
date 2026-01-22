#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2024 Pedro Guijas Bravo.
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

"""Context Information Updating commands."""

from __future__ import annotations

from p2pfl.communication.commands.command import Command


class LossInformationUpdatingCommand(Command):
    """LossInformationUpdatingCommand for AsyncDFL workflow."""

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "loss_information_updating"

    async def execute(self, source: str, round: int, loss: str, **kwargs) -> None:
        """
        Execute the command.

        Args:
            source: The source of the command.
            round: The round of the command.
            loss: Training loss of the source.
            **kwargs: The command keyword arguments.

        """
        if loss is None:
            raise ValueError("Loss is required")

        await self.workflow.loss_information_received(source, round, float(loss))


class IndexInformationUpdatingCommand(Command):
    """IndexInformationUpdatingCommand for AsyncDFL workflow."""

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "index_information_updating"

    async def execute(self, source: str, round: int, **kwargs) -> None:
        """
        Execute the command.

        Args:
            source: The source of the command.
            round: The round of the command.
            **kwargs: The command keyword arguments.

        """
        if round is None:
            raise ValueError("Index is required")

        await self.workflow.iteration_index_received(source, index=round)


class ModelInformationUpdatingCommand(Command):
    """ModelInformationUpdatingCommand for AsyncDFL workflow."""

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "model_information_updating"

    async def execute(
        self,
        source: str,
        round: int,
        weights: bytes | None = None,
        contributors: list[str] | None = None,
        num_samples: int | None = None,
        **kwargs,
    ) -> None:
        """
        Execute the command.

        Args:
            source: The source of the command.
            round: The round of the command.
            weights: The model weights bytes.
            contributors: List of contributor nodes.
            num_samples: Number of samples used for training.
            **kwargs: The command keyword arguments.

        """
        if weights is None or contributors is None or num_samples is None:
            raise ValueError("Weights, contributors and weight are required")

        await self.workflow.model_received(source, round, weights, num_samples, list(contributors))


class PushSumWeightInformationUpdatingCommand(Command):
    """PushSumWeightInformationUpdatingCommand for AsyncDFL workflow."""

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "push_sum_weight_information_updating"

    async def execute(self, source: str, round: int, push_sum_weight: str, **kwargs) -> None:
        """
        Execute the command.

        Args:
            source: The source of the command.
            round: The round of the command.
            push_sum_weight: Push-sum weight of the source.
            **kwargs: The command keyword arguments.

        """
        if push_sum_weight is None:
            raise ValueError("Push-sum weight is required")

        await self.workflow.push_sum_weight_received(source, push_sum_weight=float(push_sum_weight))
