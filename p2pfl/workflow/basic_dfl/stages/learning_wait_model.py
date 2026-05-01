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
"""Wait-for-model stage for non-training nodes in BasicDFL."""

from __future__ import annotations

from p2pfl.learning.frameworks.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.workflow.basic_dfl.context import BasicDFLContext
from p2pfl.workflow.engine.message import on_message
from p2pfl.workflow.engine.stage import Stage
from p2pfl.workflow.shared.utils import wait_with_timeout


class LearningWaitModelStage(Stage[BasicDFLContext]):
    """Non-training nodes wait here for a full model from trainers."""

    async def run(self) -> str | None:
        """Wait for full model, then skip to next round."""
        ctx = self.ctx

        await wait_with_timeout(
            ctx.full_model_ready,
            Settings.training.AGGREGATION_TIMEOUT,
            ctx.address,
            "Timeout waiting for full model. Proceeding anyway.",
        )
        ctx.needs_full_model = False
        logger.info(ctx.address, f"✅ Round {ctx.experiment.round} finished.")
        ctx.experiment.round += 1
        return "round_init"

    @on_message("add_model", weights=True, during={"learning_wait_model"})
    async def handle_add_model(
        self,
        source: str,
        round: int,
        weights: bytes,
        contributors: list[str] | None,
        num_samples: int | None,
    ) -> None:
        """Handle an add_model message containing a full model from a peer."""
        ctx = self.ctx
        if round < ctx.experiment.round:
            logger.warning(ctx.address, f"⚠️ Ignoring stale add_model from {source} (round {round}, local {ctx.experiment.round})")
            return
        try:
            logger.info(ctx.address, "📥 Full model received.")
            ctx.learner.set_model(weights)
            ctx.full_model_ready.set()
        except DecodingParamsError:
            logger.error(ctx.address, "❌ Error decoding parameters.")
        except ModelNotMatchingError:
            logger.error(ctx.address, "❌ Models not matching.")
        except Exception as e:
            logger.error(ctx.address, f"❌ Unknown error adding model: {e}")
