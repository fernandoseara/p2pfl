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
"""Workflow module for federated learning workflows."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

from .factory import WorkflowType, create_workflow

if TYPE_CHECKING:
    from p2pfl.workflow.engine.workflow import Workflow


class CommandEntry(NamedTuple):
    """Registry entry for an @on_message-decorated method."""

    method_name: str
    is_weights: bool


class on_message:
    """
    Decorator to mark a workflow method as a command handler.

    The Workflow base class auto-discovers decorated methods and registers
    them as communication commands (MessageCommand or WeightsCommand).
    They are unregistered when the workflow is cleaned up.

    A class-level ``_message_registry`` dict is built automatically
    at class definition time via ``__set_name__``.  Use
    ``Workflow.get_message_registry()`` to obtain the merged registry.

    Args:
        name: The command name used for message routing.
        weights: If True, the handler receives weight-specific kwargs
            (weights, contributors, num_samples) via WeightsCommand.
            If False (default), receives string args via MessageCommand.

    Example::

        class MyWorkflow(Workflow):
            @on_message("vote_train_set")
            async def handle_vote(self, source, round, *args):
                ...

            @on_message("partial_model", weights=True)
            async def handle_partial(self, source, round, weights, contributors, num_samples):
                ...

    """

    command_name: str
    is_weights: bool

    def __init__(self, name: str, weights: bool = False) -> None:
        """Initialize the decorator with a command name and optional weights flag."""
        self.command_name = name
        self.is_weights = weights
        self._func: Callable[..., Any] | None = None

    def __call__(self, func: Callable[..., Any]) -> on_message:
        """Wrap the decorated function."""
        self._func = func
        functools.update_wrapper(self, func)
        return self

    def __set_name__(self, owner: type[Workflow], name: str) -> None:
        """Register this command in the owner class's registry."""
        # Ensure each subclass has its own registry (don't mutate parent's)
        if "_message_registry" not in owner.__dict__:
            owner._message_registry = {}
        registry: dict[str, CommandEntry] = owner._message_registry
        registry[self.command_name] = CommandEntry(
            method_name=name,
            is_weights=self.is_weights,
        )

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        """Descriptor protocol: bind the wrapped function to the instance."""
        if obj is None:
            return self
        if self._func is None:
            raise RuntimeError("on_message decorator not properly initialized")
        return self._func.__get__(obj, objtype)


__all__ = [
    "CommandEntry",
    "WorkflowType",
    "create_workflow",
    "on_message",
]
