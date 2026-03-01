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
"""Workflow base class."""

from __future__ import annotations

import asyncio
import contextlib
import enum
import random
from abc import abstractmethod
from collections.abc import Callable
from difflib import get_close_matches
from typing import TYPE_CHECKING, Any, Generic

from p2pfl.management.logger import logger
from p2pfl.workflow.engine.context import TContext
from p2pfl.workflow.engine.experiment import Experiment
from p2pfl.workflow.engine.message import MessageEntry
from p2pfl.workflow.engine.stage import Stage
from p2pfl.workflow.validation import validate_workflow

if TYPE_CHECKING:
    from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
    from p2pfl.learning.aggregators.aggregator import Aggregator
    from p2pfl.learning.frameworks.learner import Learner


class WorkflowStatus(enum.Enum):
    """Status of a workflow run."""

    IDLE = "idle"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        """Check if the workflow reached any conclusive state."""
        return self in (WorkflowStatus.FINISHED, WorkflowStatus.CANCELLED, WorkflowStatus.FAILED)


class Workflow(Generic[TContext]):
    """
    Base class for learning workflows.

    Subclasses must implement:
    - ``initial_stage``: class attribute or property with the first stage name
    - ``get_stages()``: returns ``list[Stage[TContext]]``
    - ``create_context(**kwargs)``: builds the typed context from run kwargs

    Stage names are derived automatically from each stage class (see
    ``Stage.__init_subclass__``).  Override ``Stage.name`` as a class
    attribute to customize.

    Example::

        class BasicDFL(Workflow[BasicDFLContext]):
            initial_stage = "setup"
            context_class = BasicDFLContext

            def get_stages(self) -> list[Stage[BasicDFLContext]]:
                return [SetupStage(), VotingStage(), LearningStage(), FinishStage()]
    """

    _message_registry: dict[str, MessageEntry]

    initial_stage: str
    context_class: type[TContext]

    def __init__(self) -> None:
        """Initialize the workflow."""
        self.status: WorkflowStatus = WorkflowStatus.IDLE
        self.error: Exception | None = None
        self._task: asyncio.Task[Experiment] | None = None
        self._stage_map: dict[str, Stage[TContext]] = {}
        self._current_stage: Stage[TContext] | None = None
        self._handlers: dict[str, list[tuple[Callable[..., Any], MessageEntry]]] = {}

    ############################
    #    Abstract interface    #
    ############################

    @abstractmethod
    def get_stages(self) -> list[Stage[TContext]]:
        """Return the list of stages for this workflow."""
        ...

    def create_context(
        self,
        address: str,
        learner: Learner,
        aggregator: Aggregator,
        cp: CommunicationProtocol,
        generator: random.Random,
        experiment: Experiment,
        **kwargs: Any,
    ) -> TContext:
        """
        Build the typed context from run parameters.

        Uses ``context_class`` to construct the context with the base fields
        plus any workflow-specific kwargs. Override for custom initialization.
        """
        return self.context_class(
            address=address,
            learner=learner,
            aggregator=aggregator,
            cp=cp,
            generator=generator,
            experiment=experiment,
            **kwargs,
        )

    #####################
    #    Composition    #
    #####################

    def _compose(self, ctx: TContext) -> None:
        """Wire stages, build handler map, and validate the graph."""
        self._stage_map = {s.name: s for s in self.get_stages()}
        self._handlers.clear()

        self._validate_graph()

        self._bind_context(ctx)
        self._collect_handlers()
        self._validate_during_names()

    def _bind_context(self, ctx: TContext) -> None:
        """Set the workflow context on each stage."""
        for stage in self._stage_map.values():
            stage.ctx = ctx

    def _validate_graph(self) -> None:
        """Validate the workflow stage graph via AST inspection."""
        result = validate_workflow(self._stage_map, self.initial_stage)
        if not result.is_valid:
            errors_str = "\n".join(f"  - {e}" for e in result.errors)
            raise ValueError(f"Invalid workflow graph:\n{errors_str}")

    def _collect_handlers(self) -> None:
        """Collect @on_message handlers from stages and workflow, storing bound callables."""
        for stage in self._stage_map.values():
            for cls in type(stage).__mro__:
                if cls is Stage or cls is object:
                    break
                for msg_name, entry in cls.__dict__.get("_message_registry", {}).items():
                    # Default stage handlers to their own stage if `during` not specified
                    if entry.during is None:
                        entry = MessageEntry(entry.method_name, entry.is_weights, frozenset({stage.name}))
                    self._register_handler(getattr(stage, entry.method_name), msg_name, entry)

        for cls in type(self).__mro__:
            if cls is Workflow or cls is object:
                break
            for msg_name, entry in cls.__dict__.get("_message_registry", {}).items():
                if msg_name not in self._handlers:
                    self._handlers[msg_name] = [(getattr(self, entry.method_name), entry)]

    def _register_handler(self, callback: Callable[..., Any], msg_name: str, entry: MessageEntry) -> None:
        """Register a handler, checking for collisions with overlapping ``during`` sets."""
        if msg_name in self._handlers:
            for existing_cb, existing_entry in self._handlers[msg_name]:
                if existing_entry.during is None or entry.during is None or existing_entry.during & entry.during:
                    existing_owner = type(getattr(existing_cb, "__self__", existing_cb)).__name__
                    new_owner = type(getattr(callback, "__self__", callback)).__name__
                    raise ValueError(
                        f"Handler collision: message '{msg_name}' is registered on both "
                        f"{existing_owner} and {new_owner} "
                        f"with overlapping or unscoped `during` sets. "
                        f"Use non-overlapping `during` to scope handlers to specific stages, "
                        f"or move the handler to the Workflow class."
                    )
            self._handlers[msg_name].append((callback, entry))
        else:
            self._handlers[msg_name] = [(callback, entry)]

    def _validate_during_names(self) -> None:
        """Check that all ``during`` stage names in handlers reference existing stages."""
        available_stages = sorted(self._stage_map.keys())
        errors: list[str] = []
        for msg_name, entries in self._handlers.items():
            for _, entry in entries:
                if entry.during is not None:
                    for bad in sorted(entry.during - self._stage_map.keys()):
                        suggestions = get_close_matches(bad, self._stage_map.keys(), n=1)
                        hint = f" Did you mean '{suggestions[0]}'?" if suggestions else ""
                        errors.append(
                            f"Handler '{msg_name}' has `during={{'{bad}'}}` but '{bad}' "
                            f"is not a valid stage. Available: {', '.join(available_stages)}.{hint}"
                        )
        if errors:
            raise ValueError("\n".join(errors))

    #############
    #    Run    #
    #############

    async def run(
        self,
        experiment: Experiment,
        address: str,
        learner: Learner,
        aggregator: Aggregator,
        cp: CommunicationProtocol,
        generator: random.Random,
        **kwargs: Any,
    ) -> Experiment:
        """
        Run the workflow with an explicit Experiment and context kwargs.

        The caller is responsible for constructing the ``Experiment`` instance.
        Base context fields are passed as typed parameters; any workflow-specific
        kwargs are forwarded to ``create_context()``.

        Args:
            experiment: A fully constructed Experiment describing this run.
            address: The node's network address.
            learner: The learner instance for training.
            aggregator: The aggregator instance for model aggregation.
            cp: The communication protocol for network operations.
            generator: Random number generator for reproducibility.
            **kwargs: Workflow-specific parameters forwarded to ``create_context()``.

        Returns:
            The Experiment with tracked data after completion.

        """
        self.error = None

        # 1. Build typed context
        ctx = self.create_context(
            address=address,
            learner=learner,
            aggregator=aggregator,
            cp=cp,
            generator=generator,
            experiment=experiment,
            **kwargs,
        )

        # 2. Compose stages, wire context, build handler map
        self._compose(ctx)

        # 3. Set epochs on learner
        ctx.learner.set_epochs(experiment.epochs_per_round)

        # 4. Execute stage loop
        self.status = WorkflowStatus.RUNNING
        logger.experiment_started(ctx.address, experiment)
        try:
            await self._run(ctx)
            self.status = WorkflowStatus.FINISHED
            logger.info(ctx.address, "🏁 Learning finished.")
        except asyncio.CancelledError:
            if self.status != WorkflowStatus.FINISHED:
                self.status = WorkflowStatus.CANCELLED
            logger.info(ctx.address, "🛑 Learning cancelled.")
            raise
        except Exception as e:
            self.status = WorkflowStatus.FAILED
            self.error = e
            logger.error(ctx.address, f"Learning failed: {e}")
            raise

        return ctx.experiment

    async def _run(self, ctx: TContext) -> None:
        """Run the workflow as a sequential stage loop."""
        stage_name: str | None = self.initial_stage
        while stage_name is not None:
            stage = self._stage_map.get(stage_name)
            if stage is None:
                available = ", ".join(sorted(self._stage_map.keys()))
                suggestions = get_close_matches(stage_name, self._stage_map.keys(), n=1)
                hint = f" Did you mean '{suggestions[0]}'?" if suggestions else ""
                raise ValueError(f"Unknown stage: '{stage_name}'. Available: {available}.{hint}")
            self._current_stage = stage
            logger.info(ctx.address, f"Entering stage: {stage_name}")
            stage_name = await stage.run()
        self._current_stage = None

    #########################
    #    Task Management    #
    #########################

    async def start(self, *args: Any, **kwargs: Any) -> None:
        """
        Launch the workflow as a background ``asyncio.Task``.

        Takes the same arguments as ``run()``.

        Raises:
            RuntimeError: If a workflow task is already running.

        """
        if self._task is not None and not self._task.done():
            raise RuntimeError("Workflow is already running")
        self._task = asyncio.create_task(self.run(*args, **kwargs))

    async def stop(self) -> None:
        """
        Cancel the workflow task and wait for it to finish.

        Safe to call when no task is running (no-op), when the task is
        already done (retrieves the exception to suppress the asyncio
        'exception was never retrieved' warning), or when actively running.
        """
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        elif self._task is not None and self._task.done():
            # Retrieve exception to suppress asyncio "exception never retrieved" warning
            if not self._task.cancelled():
                with contextlib.suppress(Exception):
                    self._task.exception()
        self._task = None

    async def wait(self) -> Experiment:
        """
        Await workflow completion and return the experiment.

        Propagates any exception raised during the workflow run.

        Returns:
            The Experiment with tracked data after completion.

        Raises:
            RuntimeError: If the workflow has not been started.

        """
        if self._task is None:
            raise RuntimeError("Workflow not started")
        return await self._task

    ####################
    #    Properties    #
    ####################

    @property
    def current_stage_name(self) -> str | None:
        """Get the name of the currently executing stage, or ``None`` if idle."""
        return self._current_stage.name if self._current_stage is not None else None

    @property
    def experiment(self) -> Experiment | None:
        """Get the experiment from the current stage's context, or ``None`` if idle."""
        return self._current_stage.ctx.experiment if self._current_stage is not None else None

    ##############################
    #    Message Registration    #
    ##############################

    def get_messages(self) -> dict[str, MessageEntry]:
        """
        Get all declared message entries.

        After ``_compose()``, returns entries from the live handler map.
        Before ``_compose()``, scans class-level registries (safe to call early).
        """
        if self._handlers:
            return {name: items[0][1] for name, items in self._handlers.items()}

        # Pre-compose fallback: scan class registries
        result: dict[str, MessageEntry] = {}
        for stage in (self._stage_map or {s.name: s for s in self.get_stages()}).values():
            for cls in type(stage).__mro__:
                if cls is Stage or cls is object:
                    break
                for msg_name, entry in cls.__dict__.get("_message_registry", {}).items():
                    result.setdefault(msg_name, entry)
        for cls in type(self).__mro__:
            if cls is Workflow or cls is object:
                break
            for msg_name, entry in cls.__dict__.get("_message_registry", {}).items():
                result.setdefault(msg_name, entry)
        return result
