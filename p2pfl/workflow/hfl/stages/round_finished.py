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
"""Round finished stage for HFL (shared by workers and edges)."""

from __future__ import annotations

from p2pfl.management.logger import logger
from p2pfl.workflow.engine.stage import Stage
from p2pfl.workflow.hfl.context import HFLContext
from p2pfl.workflow.shared.evaluate import evaluate_and_broadcast


class HFLRoundFinishedStage(Stage[HFLContext]):
    """Round completion -- shared by workers and edges."""

    name = "round_finished"

    async def run(self) -> str | None:
        """Reset peers, advance round, and branch or finish."""
        ctx = self.ctx
        address = ctx.address

        # Reset peer state for next round
        for peer in ctx.peers.values():
            peer.reset_round()

        # Advance round
        ctx.experiment.increase_round(address)
        logger.info(address, f"Round {ctx.experiment.round} finished.")

        # Check if more rounds remain
        if not ctx.experiment.is_complete():
            if ctx.role == "worker":
                return "worker_train"
            return "edge_local_train"

        # Final evaluation
        await evaluate_and_broadcast(ctx)
        return None
