#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2022 Pedro Guijas Bravo.
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
"""PyTransitions utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from transitions.extensions import HierarchicalAsyncMachine
from transitions.extensions.asyncio import AsyncTimeout
from transitions.extensions.states import add_state_features

Callback = str | Callable[..., None]
Action = str | list[str]


@add_state_features(AsyncTimeout)
class TimeoutMachine(HierarchicalAsyncMachine):
    """State machine with timeout support."""

    pass

@dataclass
class StateAdapter:
    """State adapter for defining states."""

    name: str

    # Lifecycle hooks
    on_enter: Callback|None = None
    on_exit: Callback|None = None
    on_final: Callback|None = None
    on_timeout: Callback|None = None

    # State behavior
    timeout: float|None = None
    initial: str|None = None
    final: bool = False

    # Hierarchy
    children: list["StateAdapter"] = field(default_factory=list)
    parent: "StateAdapter"|None = field(default=None, repr=False)

    def is_composite(self) -> bool:
        """Check if the state is composite."""
        return bool(self.children)

    def get_child(self, name: str) -> "StateAdapter"|None:
        """Get a child state by name."""
        for child in self.children:
            if child.name == name:
                return child
        return None

    def set_parent_for_children(self):
        """Set parent for all children recursively."""
        for child in self.children:
            child.parent = self
            child.set_parent_for_children()

    def to_dict(self) -> dict[str, Any]:
        """Convert the state to a dictionary."""
        state_dict: dict[str, Any] = {
            "name": self.name,
        }

        if self.on_enter is not None:
            state_dict["on_enter"] = self.on_enter
        if self.on_exit is not None:
            state_dict["on_exit"] = self.on_exit
        if self.on_final is not None:
            state_dict["on_final"] = self.on_final
        if self.on_timeout is not None:
            state_dict["on_timeout"] = self.on_timeout
        if self.timeout is not None:
            state_dict["timeout"] = self.timeout
        if self.initial is not None:
            state_dict["initial"] = self.initial
        if self.final:
            state_dict["final"] = True
        if self.children:
            state_dict["children"] = [child.to_dict() for child in self.children]

        return state_dict

@dataclass(frozen=True)
class TransitionAdapter:
    """Transition adapter for defining transitions."""

    trigger: str
    source: str
    dest: str
    conditions: Action | None = None
    prepare: Action | None = None
    before: Action | None = None
    after: Action | None = None

    def __post_init__(self):
        """Normalize fields after initialization."""
        object.__setattr__(self, "conditions", self._normalize(self.conditions))
        object.__setattr__(self, "prepare", self._normalize(self.prepare))
        object.__setattr__(self, "before", self._normalize(self.before))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransitionAdapter":
        """
        Create a Transition instance from a dictionary.

        Args:
            data: A dictionary containing transition attributes.

        Returns:
            An instance of Transition.

        """
        return cls(
            trigger=data["trigger"],
            source=data["source"],
            dest=data["dest"],
            conditions=data.get("conditions"),
            prepare=data.get("prepare"),
            before=data.get("before"),
        )

    def _normalize(self, value: Action | None) -> list[str] | None:
        """Normalize an Action to a list of strings."""
        if value is None:
            return None
        if isinstance(value, str):
            return [value]
        return list(value)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the Transition instance back into a dictionary, omitting empty or None fields.

        Returns:
            A dictionary representation of the Transition.

        """
        data: dict[str, Any] = {
            "trigger": self.trigger,
            "source": self.source,
            "dest": self.dest,
        }

        if self.conditions:
            data["conditions"] = (
                self.conditions[0] if len(self.conditions) == 1 else self.conditions
            )

        if self.prepare:
            data["prepare"] = (
                self.prepare[0] if len(self.prepare) == 1 else self.prepare
            )

        if self.before:
            data["before"] = (
                self.before[0] if len(self.before) == 1 else self.before
            )
        return data
