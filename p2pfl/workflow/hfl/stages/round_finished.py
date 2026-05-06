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
        """Reset peers, advance round, optionally save root samples/metrics, and branch or finish."""
        ctx = self.ctx
        address = ctx.address

        # Reset peer state for next round
        for peer in ctx.peers.values():
            peer.reset_round()

        # Advance round
        ctx.experiment.round += 1
        current_round = ctx.experiment.round
        logger.info(address, f"Round {current_round} finished.")

        # Root-only: every K rounds, generate samples and log distribution metrics if the model supports it
        if ctx.role == "root":
            self._maybe_save_round_artifacts(ctx, current_round)

        # Check if more rounds remain
        if not ctx.experiment.is_complete():
            if ctx.role == "worker":
                return "worker_train"
            if ctx.role == "root":
                return "root_aggregate"
            if ctx.edge_trains:
                return "edge_local_train"
            return "edge_aggregate_workers"

        # Final evaluation
        await evaluate_and_broadcast(ctx)
        return None

    def _maybe_save_round_artifacts(self, ctx: HFLContext, current_round: int) -> None:
        """If the model exposes diffusion hooks, save per-class samples and log W2/MMD metrics."""
        every = getattr(ctx.experiment, "save_samples_every", 0) or 0
        run_dir = getattr(ctx.experiment, "run_dir", None)
        if every <= 0 or run_dir is None or current_round % every != 0:
            return

        underlying = getattr(ctx.learner.get_model(), "model", None)
        if underlying is None:
            return

        if hasattr(underlying, "evaluate_distribution_metrics"):
            try:
                metrics = underlying.evaluate_distribution_metrics()
                for name, value in metrics.items():
                    logger.log_metric(ctx.address, name, float(value), round=current_round)
            except Exception as exc:
                logger.warning(ctx.address, f"Could not compute distribution metrics: {exc}")

        if hasattr(underlying, "save_round_samples"):
            try:
                underlying.save_round_samples(current_round, run_dir)
            except Exception as exc:
                logger.warning(ctx.address, f"Could not save round samples: {exc}")
