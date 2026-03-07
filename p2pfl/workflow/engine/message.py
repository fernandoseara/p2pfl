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
"""Message descriptor for workflow stages."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, NamedTuple


class MessageEntry(NamedTuple):
    """Registry entry for an @on_message-decorated method."""

    method_name: str
    is_weights: bool
    during: frozenset[str] | None = None


class OnMessage:
    """
    Register a stage method as a message handler.

    A class-level ``_message_registry`` dict is built automatically
    at class definition time via ``__set_name__``.

    Args:
        name: The message name used for routing.
        weights: If True, the handler receives weight-specific kwargs
            (weights, contributors, num_samples) via WorkflowCommand.
        during: Optional set of stage names during which this handler is
            active. Defaults to the stage's own name.
            Set explicitly to listen during multiple stages.

    Example::

        class MyStage(Stage[MyContext]):
            @on_message("vote_train_set")  # active only during this stage
            async def handle_vote(self, source, round, *args):
                ...

            @on_message("partial_model", weights=True)
            async def handle_partial(self, source, round, weights, contributors, num_samples):
                ...

    """

    def __init__(self, name: str, weights: bool = False, during: set[str] | None = None) -> None:
        """Initialize the descriptor with a message name and optional flags."""
        self.message_name = name
        self.is_weights = weights
        self.during = frozenset(during) if during is not None else None
        self._func: Callable[..., Any] | None = None

    def __call__(self, func: Callable[..., Any]) -> OnMessage:
        """Wrap the decorated function."""
        self._func = func
        functools.update_wrapper(self, func)
        return self

    def __set_name__(self, owner: type, name: str) -> None:
        """Register this message in the owner class's registry."""
        if "_message_registry" not in owner.__dict__:
            owner._message_registry = {}  # type: ignore[attr-defined]
        registry: dict[str, MessageEntry] = owner.__dict__["_message_registry"]
        registry[self.message_name] = MessageEntry(
            method_name=name,
            is_weights=self.is_weights,
            during=self.during,
        )

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        """Descriptor protocol: bind the wrapped function to the instance."""
        if obj is None:
            return self
        if self._func is None:
            raise RuntimeError("OnMessage descriptor not properly initialized")
        return self._func.__get__(obj, objtype)


on_message = OnMessage
