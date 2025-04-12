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

"""FullModelCommand."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from transitions import MachineError

from p2pfl.communication.commands.command import Command
from p2pfl.learning.frameworks.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.management.logger import logger

if TYPE_CHECKING:  # Only imports the below statements during type checking
    from p2pfl.node import Node

class FullModelCommand(Command):
    """FullModelCommand."""

    def __init__(self, node: Node) -> None:
        """Initialize FullModelCommand."""
        self.__node = node

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "add_model"

    async def execute(
        self,
        source: str,
        round: int,
        weights: Optional[bytes] = None,
        **kwargs,
    ) -> None:
        """Execute the command."""
        if weights is None:
            logger.error(self.__node.local_state.addr, "Invalid FullModelCommand message")
            return

        try:
            logger.info(self.__node.local_state.addr, "📦 Aggregated model received.")
            self.__node.learning_workflow.full_aggregated_model_received(round, weights)

        except MachineError:
            logger.debug(self.__node.local_state.addr, "😲 Aggregated model not expected.")
        except DecodingParamsError:
            logger.error(self.__node.local_state.addr, "❌ Error decoding parameters.")
        except ModelNotMatchingError:
            logger.error(self.__node.local_state.addr, "❌ Models not matching.")
        except Exception as e:
            logger.error(self.__node.local_state.addr, f"❌ Unknown error adding model: {e}")
