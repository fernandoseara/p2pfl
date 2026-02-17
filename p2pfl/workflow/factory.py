#
# This file is part of the p2pfl distribution (see https://github.com/pguijas/p2pfl).
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
"""Workflow factory - creates learning workflows."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from p2pfl.node import Node
    from p2pfl.workflow.engine.workflow import Workflow


class WorkflowType(str, Enum):
    """Workflow type enum."""

    BASIC = "basic"
    ASYNC = "async"


def create_workflow(workflow_type: WorkflowType, node: Node) -> Workflow:
    """
    Create a learning workflow for the given node.

    Args:
        workflow_type: The type of workflow to create.
        node: The node that will run this workflow.

    Returns:
        The initialized learning workflow.

    Raises:
        ValueError: If the workflow type is not recognized.

    """
    if workflow_type == WorkflowType.BASIC:
        from p2pfl.workflow.basic_dfl.workflow import BasicDFL

        return BasicDFL(node)
    elif workflow_type == WorkflowType.ASYNC:
        from p2pfl.workflow.async_dfl.workflow import AsyncDFL

        return AsyncDFL(node)
    else:
        raise ValueError(f"Unknown workflow type: {workflow_type}")
