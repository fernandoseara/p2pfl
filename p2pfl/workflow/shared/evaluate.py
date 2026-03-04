#
# This file is part of the p2pfl (see https://github.com/pguijas/p2pfl).
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
"""Shared evaluate + broadcast utility used by workflow stages."""

from __future__ import annotations

from p2pfl.communication.commands.infrastructure import MetricsCommand
from p2pfl.management.logger import logger
from p2pfl.workflow.engine.context import WorkflowContext


async def evaluate_and_broadcast(ctx: WorkflowContext) -> None:
    """Evaluate the model and broadcast metrics to peers."""
    logger.info(ctx.address, "📊 Evaluating...")
    try:
        results = await ctx.learner.evaluate()
        logger.info(ctx.address, f"Evaluated. Results: {results}")

        if len(results) > 0:
            logger.info(ctx.address, "📡 Broadcasting metrics.")
            flattened_metrics = [str(item) for pair in results.items() for item in pair]
            await ctx.cp.broadcast_gossip(
                ctx.cp.build_msg(
                    MetricsCommand.get_name(),
                    flattened_metrics,
                    round=ctx.experiment.round,
                )
            )
    except Exception as e:
        logger.error(ctx.address, f"Evaluation failed: {e}")
