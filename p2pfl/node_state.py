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


class NodeState(Enum):
    """Unified lifecycle state for a node."""

    STOPPED = "stopped"
    IDLE = "idle"
    LEARNING = "learning"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_running(self) -> bool:
        """Check if the node is running (not stopped)."""
        return self != NodeState.STOPPED

    @property
    def is_learning(self) -> bool:
        """Check if the node is actively learning."""
        return self == NodeState.LEARNING

    @property
    def is_terminal(self) -> bool:
        """Check if the node reached a terminal learning state."""
        return self in (NodeState.FINISHED, NodeState.FAILED, NodeState.CANCELLED)


@dataclass(frozen=True)
class NodeStatus:
    """Immutable snapshot of complete node status."""

    address: str
    state: NodeState
    num_neighbors: int
    round: int | None
    total_rounds: int | None
    experiment_name: str | None
    error: str | None
    workflow_state: str | None

    def __str__(self) -> str:
        """Return a one-liner summary."""
        parts = [f"NodeStatus({self.address}, state={self.state.value}"]
        if self.experiment_name is not None:
            parts.append(f", experiment={self.experiment_name}")
        if self.round is not None:
            parts.append(f", round={self.round}/{self.total_rounds}")
        if self.error is not None:
            parts.append(f", error={self.error}")
        if self.workflow_state is not None:
            parts.append(f", workflow={self.workflow_state}")
        parts.append(f", neighbors={self.num_neighbors})")
        return "".join(parts)
