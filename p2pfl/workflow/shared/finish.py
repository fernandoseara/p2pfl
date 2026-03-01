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
"""Shared finish stage for all workflows."""

from __future__ import annotations

from p2pfl.management.logger import logger
from p2pfl.workflow.engine.context import TContext
from p2pfl.workflow.engine.stage import Stage
from p2pfl.workflow.shared.evaluate import evaluate_and_broadcast


class FinishStage(Stage[TContext]):
    """Final evaluation and training completion stage."""

    async def run(self) -> str | None:
        """Perform final evaluation and signal workflow completion."""
        await evaluate_and_broadcast(self.ctx)
        logger.info(self.ctx.address, "Training finished!!")
        return None
