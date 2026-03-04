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
"""Generic stage base class for workflow engine."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Generic

from p2pfl.workflow.engine.context import TContext


class Stage(ABC, Generic[TContext]):
    """
    A single step in a workflow, generic over the context type.

    Each stage implements ``run()`` which executes the stage logic and
    returns the name of the next stage (or ``None`` to finish the workflow).

    The stage name is derived automatically from the class name by stripping
    the ``Stage`` suffix and converting to snake_case. Override ``name`` as
    a class attribute to customize::

        class MyCustomStage(Stage[Ctx]):
            name = "custom"  # overrides auto-derived "my_custom"

    Access the workflow context via ``self.ctx``, which is set by the
    workflow engine during ``_compose()`` before any stage runs.

    Example::

        class TrainStage(Stage[BasicDFLContext]):
            async def run(self) -> str | None:
                await self.ctx.learner.fit()
                return "aggregate"

            @on_message("share_model", weights=True)
            async def handle_model(self, source, round, weights, contributors, num_samples):
                self.ctx.aggregator.add(...)
    """

    name: str
    ctx: TContext

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-derive ``name`` from the class name if not explicitly set."""
        super().__init_subclass__(**kwargs)
        if "name" not in cls.__dict__:
            cls_name = cls.__name__
            if cls_name.endswith("Stage"):
                cls_name = cls_name[:-5]
            cls.name = re.sub(r"(?<!^)(?=[A-Z])", "_", cls_name).lower()

    @abstractmethod
    async def run(self) -> str | None:
        """
        Execute this stage's logic.

        Returns:
            The name of the next stage, or ``None`` to finish the workflow.

        """
        ...
