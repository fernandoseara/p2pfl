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

"""PartialModelCommand command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from transitions import MachineError

from p2pfl.communication.commands.command import Command
from p2pfl.learning.frameworks.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.management.logger import logger

if TYPE_CHECKING:
    from p2pfl.node import Node


class PartialModelCommand(Command):
    """PartialModelCommand."""

    def __init__(
        self,
        node: Node,
    ) -> None:
        """Initialize PartialModelCommand."""
        self.__node = node

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "partial_model"

    async def execute(
        self,
        source: str,
        round: int,
        weights: bytes | None = None,
        contributors: list[str] | None = None,  # TIPO ESTA MAL (NECESARIO CASTEARLO AL LLAMAR)
        num_samples: int | None = None,
        **kwargs,
    ) -> None:
        """Execute the command."""
        if weights is None or contributors is None or num_samples is None:
            raise ValueError("Weights, contributors and weight are required")

        try:
            # Add model to aggregator
            model = (
                self.__node.get_learner()
                .get_model()
                .build_copy(
                    params=weights,
                    num_samples=num_samples,
                    contributors=list(contributors),
                )
            )

            await self.__node.get_learning_workflow().aggregate(model, source)

        except MachineError:
            logger.debug(self.__node.address, "Invalid state.")
        except DecodingParamsError:
            logger.error(self.__node.address, "Error decoding parameters.")
        except ModelNotMatchingError:
            logger.error(self.__node.address, "Models not matching.")
        except Exception as e:
            logger.error(self.__node.address, f"Unknown error adding model: {e}")
