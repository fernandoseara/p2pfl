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
"""Workflow graph validation via AST inspection."""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from p2pfl.workflow.engine.stage import Stage
    from p2pfl.workflow.engine.workflow import Workflow


@dataclass
class StageTransitions:
    """Extracted transitions from a single stage."""

    stage_name: str
    targets: set[str | None] = field(default_factory=set)
    dynamic_returns: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result of workflow graph validation."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    transitions: dict[str, StageTransitions] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Return True if no errors were found."""
        return len(self.errors) == 0

    def __str__(self) -> str:
        """Format as a readable report."""
        lines: list[str] = []
        if self.errors:
            lines.append("ERRORS:")
            for e in self.errors:
                lines.append(f"  - {e}")
        if self.warnings:
            lines.append("WARNINGS:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        if not self.errors and not self.warnings:
            lines.append("Workflow graph is valid.")
        return "\n".join(lines)


def _extract_returns_from_run(stage: Stage[Any]) -> StageTransitions:
    """
    Extract all possible return values from a stage's run() method via AST.

    Handles:
    - ``return "stage_name"`` — string literal
    - ``return None`` — explicit None / bare return
    - ``return "a" if cond else "b"`` — ternary expressions
    - Warns on dynamic returns (variables, f-strings, calls)

    """
    result = StageTransitions(stage_name="")

    try:
        source = inspect.getsource(stage.run)
    except (OSError, TypeError):
        result.dynamic_returns.append("<could not inspect source>")
        return result

    source = textwrap.dedent(source)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        result.dynamic_returns.append("<could not parse source>")
        return result

    # Find the run() function definition (top-level only)
    run_func: ast.AsyncFunctionDef | ast.FunctionDef | None = None
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef) and node.name == "run":
            run_func = node
            break

    if run_func is None:
        result.dynamic_returns.append("<run() not found in AST>")
        return result

    # Walk the function body (but NOT nested function defs)
    _collect_returns(run_func, result)

    return result


def _collect_returns(node: ast.AST, result: StageTransitions) -> None:
    """Recursively collect return values, skipping nested function definitions."""
    for child in ast.iter_child_nodes(node):
        # Skip nested functions / classes — their returns aren't ours
        if isinstance(child, ast.AsyncFunctionDef | ast.FunctionDef | ast.ClassDef):
            continue

        if isinstance(child, ast.Return):
            _extract_return_value(child, result)
        else:
            _collect_returns(child, result)


def _extract_return_value(node: ast.Return, result: StageTransitions) -> None:
    """Extract the return value from a Return AST node."""
    if node.value is None:
        # bare ``return`` or ``return None`` (implicit)
        result.targets.add(None)
        return

    _extract_from_expr(node.value, result)


def _extract_from_expr(expr: ast.expr, result: StageTransitions) -> None:
    """Extract string/None values from an expression, recursively for ternaries."""
    # String or None constant
    if isinstance(expr, ast.Constant):
        if isinstance(expr.value, str):
            result.targets.add(expr.value)
        elif expr.value is None:
            result.targets.add(None)
        else:
            result.dynamic_returns.append(f"non-string constant: {expr.value!r}")
        return

    # Ternary: ``return "a" if cond else "b"``
    if isinstance(expr, ast.IfExp):
        _extract_from_expr(expr.body, result)
        _extract_from_expr(expr.orelse, result)
        return

    # Everything else is dynamic (variable, call, f-string, etc.)
    try:
        source_fragment = ast.unparse(expr)
    except Exception:
        source_fragment = "<unknown expression>"
    result.dynamic_returns.append(source_fragment)


def validate_workflow(
    stage_map: dict[str, Stage[Any]],
    initial_stage: str,
) -> ValidationResult:
    """
    Validate a workflow's stage graph.

    Checks:
    - ``initial_stage`` exists in the stage map
    - All return values from stages reference existing stages
    - All stages are reachable from ``initial_stage``
    - The workflow can terminate (at least one stage returns None)
    - Warns on dynamic returns that can't be statically checked

    Args:
        stage_map: Mapping of stage name to Stage instance.
        initial_stage: The entry-point stage name.

    Returns:
        ValidationResult with errors, warnings, and extracted transitions.

    """
    result = ValidationResult()

    # 1. Check initial_stage exists
    if initial_stage not in stage_map:
        available = ", ".join(sorted(stage_map.keys()))
        suggestions = get_close_matches(initial_stage, stage_map.keys(), n=1)
        hint = f" Did you mean '{suggestions[0]}'?" if suggestions else ""
        result.errors.append(f"initial_stage '{initial_stage}' not in stage map. " f"Available: {available}.{hint}")
        return result

    # 2. Extract transitions from all stages
    for name, stage in stage_map.items():
        transitions = _extract_returns_from_run(stage)
        transitions.stage_name = name
        result.transitions[name] = transitions

    # 3. Validate all targets exist
    for name, transitions in result.transitions.items():
        for target in transitions.targets:
            if target is not None and target not in stage_map:
                available = ", ".join(sorted(stage_map.keys()))
                suggestions = get_close_matches(target, stage_map.keys(), n=1)
                hint = f" Did you mean '{suggestions[0]}'?" if suggestions else ""
                result.errors.append(f"Stage '{name}' returns '{target}' which is not in the stage map. " f"Available: {available}.{hint}")

    # 4. Warn on dynamic returns
    for name, transitions in result.transitions.items():
        for dyn in transitions.dynamic_returns:
            result.warnings.append(f"Stage '{name}' has a dynamic return ({dyn}) — cannot validate statically.")

    # 5. Check reachability (BFS from initial_stage)
    reachable: set[str] = set()
    queue = [initial_stage]
    while queue:
        current = queue.pop(0)
        if current in reachable:
            continue
        reachable.add(current)
        stage_transitions = result.transitions.get(current)
        if stage_transitions:
            for target in stage_transitions.targets:
                if target is not None and target not in reachable:
                    queue.append(target)

    unreachable = set(stage_map.keys()) - reachable
    for name in sorted(unreachable):
        result.warnings.append(f"Stage '{name}' is unreachable from initial_stage '{initial_stage}'.")

    # 6. Check termination (at least one stage can return None)
    has_terminal = any(None in t.targets for t in result.transitions.values())
    if not has_terminal:
        result.errors.append("No stage returns None — the workflow cannot terminate.")

    return result


def validate(workflow: Workflow[Any]) -> ValidationResult:
    """
    Validate a workflow's stage graph.

    Convenience wrapper around ``validate_workflow`` that accepts a
    ``Workflow`` instance directly.

    Args:
        workflow: The workflow to validate.

    Returns:
        ValidationResult with errors, warnings, and extracted transitions.

    """
    stage_map = {s.name: s for s in workflow.get_stages()}
    return validate_workflow(stage_map, workflow.initial_stage)
