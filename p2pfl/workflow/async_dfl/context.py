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
"""Typed context and per-peer state for AsyncDFL workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from p2pfl.workflow.engine.context import WorkflowContext

if TYPE_CHECKING:
    from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


@dataclass
class AsyncPeerState:
    """Per-peer mutable state for AsyncDFL workflow."""

    round_number: int = 0
    push_sum_weight: float = 1.0
    model: P2PFLModel | None = None
    losses: dict[int, float] = field(default_factory=dict)
    push_time: int = 0
    mixing_weight: float = 1.0
    p2p_updating_idx: int = 0

    def add_loss(self, round: int, loss: float) -> None:
        """Record a loss value at the given round index."""
        self.losses[round] = loss

    def reset_round(self) -> None:
        """Reset per-round mutable state (clear cached model)."""
        self.model = None


@dataclass
class AsyncDFLContext(WorkflowContext):
    """
    Typed context for AsyncDFL workflow.

    Extends WorkflowContext with peer tracking, candidate list,
    and the tau parameter for network update frequency.
    """

    peers: dict[str, AsyncPeerState] = field(default_factory=dict)
    candidates: list[str] = field(default_factory=list)
    tau: int = 2
    dmax: int = 5
    top_k_neighbors: int = 3
