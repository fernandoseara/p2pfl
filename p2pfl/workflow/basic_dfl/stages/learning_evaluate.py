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
"""Evaluate stage for BasicDFL."""

from __future__ import annotations

from p2pfl.workflow.basic_dfl.context import BasicDFLContext
from p2pfl.workflow.engine.stage import Stage
from p2pfl.workflow.shared.evaluate import evaluate_and_broadcast


class LearningEvaluateStage(Stage[BasicDFLContext]):
    """Evaluate the current model before training."""

    async def run(self) -> str | None:
        """Evaluate and broadcast metrics, then proceed to training."""
        ctx = self.ctx
        ctx.models_complete.clear()
        ctx.full_model_ready.clear()

        if ctx.needs_full_model:
            # Non-training node: skip evaluation, go wait for full model
            return "learning_wait_model"

        await evaluate_and_broadcast(ctx)
        return "learning_train"
