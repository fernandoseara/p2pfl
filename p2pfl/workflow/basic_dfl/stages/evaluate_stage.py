#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2026 Pedro Guijas Bravo.
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
"""Evaluate stage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from p2pfl.communication.commands.infrastructure import MetricsCommand
from p2pfl.management.logger import logger
from p2pfl.workflow.engine.stage import Stage

if TYPE_CHECKING:
    from p2pfl.node import Node


class EvaluateStage(Stage):
    """Evaluate stage."""

    @staticmethod
    async def execute(node: Node) -> None:
        """Execute the stage."""
        assert node.workflow is not None
        # Evaluate and send metrics
        logger.info(node.address, "🔬 Evaluating...")
        results = await node.learner.evaluate()
        logger.info(node.address, f"📈 Evaluated. Results: {results}")

        # Send metrics
        if len(results) > 0:
            logger.info(node.address, "📢 Broadcasting metrics.")
            flattened_metrics = [str(item) for pair in results.items() for item in pair]
            await node.communication_protocol.broadcast_gossip(
                node.communication_protocol.build_msg(
                    MetricsCommand.get_name(),
                    flattened_metrics,
                    round=node.workflow.round,
                )
            )
