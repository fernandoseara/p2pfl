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
"""Train stage for BasicDFL — pure local training (trainers only)."""

from __future__ import annotations

from p2pfl.management.logger import logger
from p2pfl.workflow.basic_dfl.context import BasicDFLContext
from p2pfl.workflow.engine.stage import Stage


class LearningTrainStage(Stage[BasicDFLContext]):
    """Local model training (trainers only). Non-trainers use LearningWaitModelStage."""

    async def run(self) -> str | None:
        """Fit the model and proceed to gossip."""
        ctx = self.ctx

        await ctx.learner.fit()
        logger.info(ctx.address, "🎓 Training done.")

        # Save own model for aggregation
        peer = ctx.peers.get(ctx.address)
        if peer is not None:
            peer.model = ctx.learner.get_model()

        return "learning_gossip_loop"
