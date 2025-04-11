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
"""Initialize model stage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from p2pfl.learning.frameworks.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.management.logger import logger
from p2pfl.stages.stage import Stage

if TYPE_CHECKING:
    from p2pfl.node import Node


class InitializeModelStage(Stage):
    """Initialize model stage."""

    @staticmethod
    async def execute(
        round: int,
        weights: bytes,
        node: Node
        ) -> None:
        """Execute the stage."""
        # Check source
        if round != node.get_local_state().round:
            logger.debug(
                node.address,
                f"Model initialization in a late round ({round} != {node.get_local_state().round}).",
            )
            return

        try:
            # Set new weights
            node.get_learner().set_model(weights)

            logger.info(node.address, "🤖 Model Weights Initialized")

        except DecodingParamsError:
            logger.error(node.address, "Error decoding parameters.")

        except ModelNotMatchingError:
            logger.error(node.address, "Models not matching.")

        except Exception as e:
            logger.error(node.address, f"Unknown error adding model: {e}")
