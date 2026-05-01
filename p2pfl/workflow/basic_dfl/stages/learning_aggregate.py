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
"""Aggregate stage for BasicDFL."""

from __future__ import annotations

from typing import TYPE_CHECKING

from p2pfl.management.logger import logger
from p2pfl.workflow.basic_dfl.context import BasicDFLContext
from p2pfl.workflow.engine.stage import Stage

if TYPE_CHECKING:
    from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


class LearningAggregateStage(Stage[BasicDFLContext]):
    """Aggregate collected models and advance the round."""

    async def run(self) -> str | None:
        """Aggregate models, set the result, and advance to next round."""
        ctx = self.ctx

        aggregator = ctx.aggregator
        models = [p.model for p in ctx.peers.values() if p.model is not None]

        # Deduplicate overlapping partial aggregates to prevent double-counting
        models = self._deduplicate_models(models)

        total = len(ctx.train_set)
        contributors = {c for m in models for c in m.get_contributors()}
        if len(contributors) < total:
            logger.info(ctx.address, f"⚠️ Partial aggregation: {len(contributors)}/{total} contributors covered ({len(models)} models).")
        agg_model = aggregator.aggregate(models)
        ctx.learner.set_model(agg_model)

        logger.info(ctx.address, f"✅ Round {ctx.experiment.round} finished.")
        ctx.experiment.round += 1
        return "round_init"

    @staticmethod
    def _deduplicate_models(models: list[P2PFLModel]) -> list[P2PFLModel]:
        """
        Select non-overlapping models maximizing contributor coverage.

        Uses greedy set cover: process widest models first, skip models
        whose contributors are already fully covered by selected models.
        """
        sorted_models = sorted(models, key=lambda m: len(m.get_contributors()), reverse=True)
        covered: set[str] = set()
        result: list[P2PFLModel] = []
        for m in sorted_models:
            contribs = set(m.get_contributors())
            if contribs - covered:  # has at least one new contributor
                result.append(m)
                covered |= contribs
        return result
