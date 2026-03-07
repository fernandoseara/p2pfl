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
"""BasicDFL synchronous federated learning workflow."""

from __future__ import annotations

from p2pfl.workflow.basic_dfl.context import BasicDFLContext
from p2pfl.workflow.basic_dfl.stages import (
    FinishStage,
    LearningStage,
    RoundInitStage,
    SetupStage,
    VotingStage,
)
from p2pfl.workflow.engine.stage import Stage
from p2pfl.workflow.engine.workflow import Workflow


class BasicDFL(Workflow[BasicDFLContext]):
    """
    BasicDFL: synchronous decentralized federated learning.

    Flow: setup -> round_init -> voting -> learning -> round_init -> ... -> finish
    """

    context_class = BasicDFLContext

    def get_stages(self) -> list[Stage[BasicDFLContext]]:
        """Return the stages for BasicDFL."""
        return [SetupStage(), RoundInitStage(), VotingStage(), LearningStage(), FinishStage()]

    def validate_experiment(self, ctx: BasicDFLContext) -> None:
        """Resolve dynamic defaults and validate BasicDFL experiment params."""
        exp = ctx.experiment
        if "trainset_size" not in exp.data:
            exp.data["trainset_size"] = len(ctx.peers)
        if exp.data["trainset_size"] < 1:
            raise ValueError("trainset_size must be >= 1")
