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
"""Node state enum and status snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from p2pfl.workflow.engine.experiment import Experiment
    from p2pfl.workflow.engine.workflow import WorkflowStatus


class NodeState(Enum):
    """Unified lifecycle state for a node."""

    OFFLINE = "offline"
    IDLE = "idle"
    LEARNING = "learning"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_running(self) -> bool:
        """Check if the node is running (not stopped)."""
        return self != NodeState.OFFLINE

    @property
    def is_learning(self) -> bool:
        """Check if the node is actively learning."""
        return self == NodeState.LEARNING

    @property
    def is_terminal(self) -> bool:
        """Check if the node reached a terminal learning state."""
        return self in (NodeState.FINISHED, NodeState.FAILED, NodeState.CANCELLED)

    @classmethod
    def from_workflow_status(cls, ws: WorkflowStatus) -> NodeState:
        """Map a WorkflowStatus to the corresponding NodeState."""
        from p2pfl.workflow.engine.workflow import WorkflowStatus

        mapping = {
            WorkflowStatus.FAILED: cls.FAILED,
            WorkflowStatus.CANCELLED: cls.CANCELLED,
            WorkflowStatus.FINISHED: cls.FINISHED,
            WorkflowStatus.RUNNING: cls.LEARNING,
            WorkflowStatus.IDLE: cls.IDLE,
        }
        return mapping.get(ws, cls.IDLE)


@dataclass(frozen=True)
class NodeStatus:
    """Immutable snapshot of complete node status."""

    address: str
    state: NodeState
    num_neighbors: int
    experiment: Experiment | None
    error: str | None
    current_stage_name: str | None

    def __str__(self) -> str:
        """Return a one-liner summary."""
        parts = [f"NodeStatus({self.address}, state={self.state.value}"]
        if self.experiment is not None:
            parts.append(f", experiment={str(self.experiment)}")
        if self.error is not None:
            parts.append(f", error={self.error}")
        if self.current_stage_name is not None:
            parts.append(f", stage={self.current_stage_name}")
        parts.append(f", neighbors={self.num_neighbors})")
        return "".join(parts)
