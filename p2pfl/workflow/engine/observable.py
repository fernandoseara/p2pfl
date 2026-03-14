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
"""Observable mixin for notifying observers of attribute changes."""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from typing import Any

_SENTINEL = object()


class Observer(ABC):
    """Abstract observer that receives notifications when observed attributes change."""

    @abstractmethod
    def update(self, field_name: str, value: Any) -> None:
        """
        Handle an observed attribute change.

        Args:
            field_name: The name of the changed attribute.
            value: The new value.

        """
        ...


class Observable:
    """
    Mixin that notifies observers on ``__setattr__`` changes.

    Safe with ``@dataclass``: ``_observers`` is lazily initialized via
    ``getattr`` because dataclass-generated ``__init__`` doesn't call
    ``super().__init__()``, and a class-level mutable default would be
    shared across all instances.
    """

    def add_observer(self, observer: Observer) -> None:
        """Register an observer."""
        observers: list[Observer] = getattr(self, "_observers", [])
        if not observers:
            object.__setattr__(self, "_observers", observers)
        observers.append(observer)

    def remove_observer(self, observer: Observer) -> None:
        """Unregister an observer (no-op if not registered)."""
        observers: list[Observer] = getattr(self, "_observers", [])
        with contextlib.suppress(ValueError):
            observers.remove(observer)

    def clear_observers(self) -> None:
        """Remove all observers."""
        observers: list[Observer] = getattr(self, "_observers", [])
        observers.clear()

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute and notify observers (skips ``_``-prefixed attrs and unchanged values)."""
        old = getattr(self, name, _SENTINEL)
        object.__setattr__(self, name, value)
        if not name.startswith("_") and (old is _SENTINEL or old != value):
            for observer in getattr(self, "_observers", []):
                observer.update(name, value)
