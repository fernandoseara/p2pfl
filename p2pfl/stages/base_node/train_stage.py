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
"""Train stage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from p2pfl.management.logger import logger
from p2pfl.stages.stage import Stage

if TYPE_CHECKING:
    from p2pfl.node import Node

class TrainStage(Stage):
    """Train stage."""

    @staticmethod
    async def execute(node: Node) -> None:
        """Execute the stage."""
        # Train
        logger.info(node.address, "🏋️‍♀️ Training...")
        await node.get_learner().fit()
        logger.info(node.address, "🎓 Training done.")

        # Add model to the state
        node.get_network_state().add_model(node.get_learner().get_model(), source=node.address)
