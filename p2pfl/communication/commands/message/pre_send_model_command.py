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

"""
PRE_SEND_MODEL command.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!                                                                            !!
!!  DEPRECATED: This file is not integrated and should not be used.           !!
!!  NO ESTA METIDO                                                            !!
!!                                                                            !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""

import warnings

from p2pfl.communication.commands.command import Command
from p2pfl.stages.local_state.node_state import LocalNodeState

warnings.warn(
    "pre_send_model_command.py is not integrated.!",
    DeprecationWarning,
    stacklevel=2,
)


class PreSendModelCommand(Command):
    """Command used to notify a recipient node before a model is actually sent."""

    def __init__(self, node_state: LocalNodeState) -> None:
        """Initialize the command."""
        self.node_state = node_state

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "PRE_SEND_MODEL"

    @staticmethod
    def remove_hashed(node_state: LocalNodeState, cmd: str, hashes: list[str], round: int) -> None:
        """Remove hashes from sending_models."""
        raise NotImplementedError("This method is deprecated and should not be used.")

    async def execute(self, source: str, round: int, *args, **kwargs) -> str | None:
        """Execute the command."""
        raise NotImplementedError("This method is deprecated and should not be used.")
