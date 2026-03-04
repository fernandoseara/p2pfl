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
"""Typed context and per-peer state for BasicDFL workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from p2pfl.workflow.engine.context import WorkflowContext

if TYPE_CHECKING:
    from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


@dataclass
class BasicPeerState:
    """Per-peer mutable state for BasicDFL workflow."""

    round_number: int = 0
    model: P2PFLModel | None = None
    aggregated_from: list[str] = field(default_factory=list)
    votes: dict[str, int] = field(default_factory=dict)

    def reset_round(self) -> None:
        """Reset per-round mutable state."""
        self.model = None
        self.aggregated_from.clear()
        self.votes.clear()


@dataclass
class BasicDFLContext(WorkflowContext):
    """
    Typed context for BasicDFL workflow.

    Extends WorkflowContext with peer tracking, train set, and model state.
    """

    peers: dict[str, BasicPeerState] = field(default_factory=dict)
    train_set: list[str] = field(default_factory=list)
    needs_full_model: bool = False
