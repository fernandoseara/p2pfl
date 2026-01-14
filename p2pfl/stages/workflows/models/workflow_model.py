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

from typing import TYPE_CHECKING

from p2pfl.management.logger import logger

if TYPE_CHECKING:
    from p2pfl.node import Node


class WorkflowModel:
    """Workflow base class."""

    state: str

    def __init__(self, node: Node):
        """Initialize the workflow."""
        self.node: Node = node

    ######################
    # LOGGING CALLBACKS #
    ######################
    def finalize_logging(self, *args, **kwargs) -> None:
        """Log the current stage transition."""
        logger.debug(self.node.address, f"🏃 Running stage: {(self.state)}")

    def test(self, *args, **kwargs) -> None:
        """Test function for debugging."""
        logger.info(self.node.address, "Test function called.")
