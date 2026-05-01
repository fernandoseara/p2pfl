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

"""Tests for workflow graph validation via AST inspection."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from p2pfl.workflow.engine.stage import Stage
from p2pfl.workflow.validation import (
    StageTransitions,
    ValidationResult,
    _extract_returns_from_run,
    validate_workflow,
)

# ---------------------------------------------------------------------------
# Helper stages for testing AST extraction
# ---------------------------------------------------------------------------


class ReturnsStringStage(Stage[Any]):
    """Stage that returns a string literal."""

    name = "returns_string"

    async def run(self) -> str | None:
        """Return a string literal."""
        return "next_stage"


class ReturnsNoneStage(Stage[Any]):
    """Stage that returns None explicitly."""

    name = "returns_none"

    async def run(self) -> str | None:
        """Return None."""
        return None


class BareReturnStage(Stage[Any]):
    """Stage with a bare return statement."""

    name = "bare_return"

    async def run(self) -> str | None:
        """Bare return."""
        return


class TernaryReturnStage(Stage[Any]):
    """Stage with a ternary return expression."""

    name = "ternary_return"

    async def run(self) -> str | None:
        """Return a ternary expression."""
        return "stage_a" if True else "stage_b"  # noqa: SIM210


class DynamicReturnStage(Stage[Any]):
    """Stage that returns a variable (dynamic, can't be validated statically)."""

    name = "dynamic_return"

    async def run(self) -> str | None:
        """Return a variable (dynamic)."""
        target = "somewhere"
        return target


class NonStringConstantStage(Stage[Any]):
    """Stage that returns a non-string constant (e.g. int)."""

    name = "non_string_constant"

    async def run(self) -> str | None:
        """Return an integer constant."""
        return 42  # type: ignore[return-value]


class NestedFunctionStage(Stage[Any]):
    """Stage whose run() contains a nested function with its own return."""

    name = "nested_function"

    async def run(self) -> str | None:
        """Return from outer function, not the nested one."""

        def helper():
            return "should_be_ignored"

        helper()
        return "actual_target"


class MultiReturnStage(Stage[Any]):
    """Stage with multiple return paths."""

    name = "multi_return"

    async def run(self) -> str | None:
        """Return from one of two branches."""
        if True:  # noqa: SIM108
            return "branch_a"
        else:
            return "branch_b"


class TerminalStage(Stage[Any]):
    """Stage that always returns None (terminal)."""

    name = "terminal"

    async def run(self) -> str | None:
        """Terminate the workflow."""
        return None


class NonTerminalStage(Stage[Any]):
    """Stage that always returns a string (non-terminal)."""

    name = "non_terminal"

    async def run(self) -> str | None:
        """Loop back to self."""
        return "non_terminal"


# ---------------------------------------------------------------------------
# Tests: AST return extraction
# ---------------------------------------------------------------------------


class TestExtractReturnsFromRun:
    """Tests for _extract_returns_from_run AST extraction."""

    def test_string_literal_return(self):
        """Extracts string literal from a return statement."""
        stage = ReturnsStringStage()
        result = _extract_returns_from_run(stage)
        assert "next_stage" in result.targets
        assert not result.dynamic_returns

    def test_none_return(self):
        """Extracts None from 'return None'."""
        stage = ReturnsNoneStage()
        result = _extract_returns_from_run(stage)
        assert None in result.targets
        assert not result.dynamic_returns

    def test_bare_return(self):
        """Extracts None from bare 'return' statement."""
        stage = BareReturnStage()
        result = _extract_returns_from_run(stage)
        assert None in result.targets

    def test_ternary_expression(self):
        """Extracts both branches of a ternary return."""
        stage = TernaryReturnStage()
        result = _extract_returns_from_run(stage)
        assert "stage_a" in result.targets
        assert "stage_b" in result.targets

    def test_dynamic_return_flagged(self):
        """Dynamic returns (variables) are recorded in dynamic_returns."""
        stage = DynamicReturnStage()
        result = _extract_returns_from_run(stage)
        assert len(result.dynamic_returns) > 0
        assert any("target" in d for d in result.dynamic_returns)

    def test_non_string_constant(self):
        """Non-string constants (e.g. int) are flagged as dynamic returns."""
        stage = NonStringConstantStage()
        result = _extract_returns_from_run(stage)
        assert len(result.dynamic_returns) > 0
        assert any("42" in d for d in result.dynamic_returns)

    def test_nested_function_returns_ignored(self):
        """Returns from nested functions are not collected."""
        stage = NestedFunctionStage()
        result = _extract_returns_from_run(stage)
        assert "should_be_ignored" not in result.targets
        assert "actual_target" in result.targets

    def test_multiple_return_branches(self):
        """All return branches are extracted."""
        stage = MultiReturnStage()
        result = _extract_returns_from_run(stage)
        assert "branch_a" in result.targets
        assert "branch_b" in result.targets

    def test_source_inspection_failure(self):
        """Handles OSError/TypeError when getsource fails."""
        stage = ReturnsStringStage()
        with patch("p2pfl.workflow.validation.inspect.getsource", side_effect=OSError("no source")):
            result = _extract_returns_from_run(stage)
        assert len(result.dynamic_returns) == 1
        assert "<could not inspect source>" in result.dynamic_returns[0]

    def test_ast_parse_failure(self):
        """Handles SyntaxError when AST parsing fails."""
        stage = ReturnsStringStage()
        with patch("p2pfl.workflow.validation.inspect.getsource", return_value="def run(self)\n    pass"):
            result = _extract_returns_from_run(stage)
        assert len(result.dynamic_returns) == 1
        assert "<could not parse source>" in result.dynamic_returns[0]

    def test_run_not_found_in_ast(self):
        """Handles case when run() function not found in parsed AST."""
        stage = ReturnsStringStage()
        # Return valid Python that has no function named "run"
        with patch("p2pfl.workflow.validation.inspect.getsource", return_value="x = 42\n"):
            result = _extract_returns_from_run(stage)
        assert len(result.dynamic_returns) == 1
        assert "<run() not found in AST>" in result.dynamic_returns[0]


# ---------------------------------------------------------------------------
# Tests: validate_workflow
# ---------------------------------------------------------------------------


class TestValidateWorkflow:
    """Tests for validate_workflow graph validation."""

    def test_valid_linear_workflow(self):
        """A simple valid linear workflow passes validation."""
        stage_map: dict[str, Stage[Any]] = {
            "start": ReturnsStringStage(),
            "terminal": TerminalStage(),
        }
        # Patch start to return "terminal"
        stage_map["start"].name = "start"  # type: ignore[misc]
        stage_map["terminal"].name = "terminal"  # type: ignore[misc]

        # ReturnsStringStage returns "next_stage", so we need a matching target
        # Instead, build a proper chain:
        class StartStage(Stage[Any]):
            name = "start"

            async def run(self) -> str | None:
                return "terminal"

        stage_map = {
            "start": StartStage(),
            "terminal": TerminalStage(),
        }
        result = validate_workflow(stage_map, "start")
        assert result.is_valid, f"Unexpected errors: {result.errors}"

    def test_initial_stage_not_in_map(self):
        """Error when initial_stage is not in the stage map."""
        stage_map: dict[str, Stage[Any]] = {
            "start": ReturnsStringStage(),
        }
        result = validate_workflow(stage_map, "nonexistent")
        assert not result.is_valid
        assert any("nonexistent" in e and "not in stage map" in e for e in result.errors)

    def test_initial_stage_suggestion_hint(self):
        """Suggestion hint when initial_stage is close to an existing name."""
        stage_map: dict[str, Stage[Any]] = {
            "setup": TerminalStage(),
        }
        result = validate_workflow(stage_map, "setpu")  # typo
        assert not result.is_valid
        assert any("Did you mean 'setup'" in e for e in result.errors)

    def test_invalid_transition_target(self):
        """Error when a stage returns a name that doesn't exist in the map."""

        class BadTransitionStage(Stage[Any]):
            name = "start"

            async def run(self) -> str | None:
                return "nonexistent_stage"

        stage_map: dict[str, Stage[Any]] = {
            "start": BadTransitionStage(),
        }
        result = validate_workflow(stage_map, "start")
        assert not result.is_valid
        assert any("nonexistent_stage" in e and "not in the stage map" in e for e in result.errors)

    def test_invalid_transition_with_suggestion(self):
        """Suggestion hint when a return target is close to an existing stage name."""

        class TypoTransitionStage(Stage[Any]):
            name = "start"

            async def run(self) -> str | None:
                return "termnial"  # typo for "terminal"

        stage_map: dict[str, Stage[Any]] = {
            "start": TypoTransitionStage(),
            "terminal": TerminalStage(),
        }
        result = validate_workflow(stage_map, "start")
        assert not result.is_valid
        assert any("Did you mean 'terminal'" in e for e in result.errors)

    def test_unreachable_stage(self):
        """Warning when a stage is unreachable from initial_stage."""

        class IsolatedStage(Stage[Any]):
            name = "isolated"

            async def run(self) -> str | None:
                return None

        class StartToTerminalStage(Stage[Any]):
            name = "start"

            async def run(self) -> str | None:
                return "terminal"

        stage_map: dict[str, Stage[Any]] = {
            "start": StartToTerminalStage(),
            "terminal": TerminalStage(),
            "isolated": IsolatedStage(),
        }
        result = validate_workflow(stage_map, "start")
        assert result.is_valid  # unreachable is a warning, not an error
        assert any("isolated" in w and "unreachable" in w for w in result.warnings)

    def test_no_terminal_stage_error(self):
        """Error when no stage can return None (workflow can't terminate)."""
        stage_map: dict[str, Stage[Any]] = {
            "non_terminal": NonTerminalStage(),
        }
        result = validate_workflow(stage_map, "non_terminal")
        assert not result.is_valid
        assert any("cannot terminate" in e for e in result.errors)

    def test_dynamic_return_warning(self):
        """Warning for stages with dynamic returns."""
        stage_map: dict[str, Stage[Any]] = {
            "dynamic_return": DynamicReturnStage(),
            "terminal": TerminalStage(),
        }
        result = validate_workflow(stage_map, "dynamic_return")
        assert any("dynamic return" in w for w in result.warnings)

    def test_cycle_does_not_cause_infinite_loop(self):
        """BFS reachability handles cycles without hanging."""

        class CycleAStage(Stage[Any]):
            name = "cycle_a"

            async def run(self) -> str | None:
                return "cycle_b"

        class CycleBStage(Stage[Any]):
            name = "cycle_b"

            async def run(self) -> str | None:
                return "cycle_a"

        stage_map: dict[str, Stage[Any]] = {
            "cycle_a": CycleAStage(),
            "cycle_b": CycleBStage(),
        }
        result = validate_workflow(stage_map, "cycle_a")
        # Both stages reachable, but no terminal → error
        assert not result.is_valid
        assert any("cannot terminate" in e for e in result.errors)
        # No unreachable warnings since both are reachable via the cycle
        assert not any("unreachable" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Tests: ValidationResult formatting
# ---------------------------------------------------------------------------


class TestValidationResult:
    """Tests for ValidationResult string representation."""

    def test_str_with_errors(self):
        """String representation includes errors."""
        vr = ValidationResult(errors=["Something is wrong"])
        s = str(vr)
        assert "ERRORS:" in s
        assert "Something is wrong" in s

    def test_str_with_warnings(self):
        """String representation includes warnings."""
        vr = ValidationResult(warnings=["Watch out"])
        s = str(vr)
        assert "WARNINGS:" in s
        assert "Watch out" in s

    def test_str_with_errors_and_warnings(self):
        """String representation includes both errors and warnings."""
        vr = ValidationResult(errors=["Bad"], warnings=["Meh"])
        s = str(vr)
        assert "ERRORS:" in s
        assert "Bad" in s
        assert "WARNINGS:" in s
        assert "Meh" in s

    def test_str_valid(self):
        """String representation for a valid result."""
        vr = ValidationResult()
        s = str(vr)
        assert "Workflow graph is valid." in s

    def test_is_valid_property(self):
        """is_valid returns True when no errors."""
        assert ValidationResult().is_valid
        assert not ValidationResult(errors=["err"]).is_valid

    def test_str_only_warnings_no_valid_message(self):
        """When there are warnings but no errors, 'valid' message is not shown."""
        vr = ValidationResult(warnings=["something"])
        s = str(vr)
        assert "Workflow graph is valid." not in s


# ---------------------------------------------------------------------------
# Tests: StageTransitions dataclass
# ---------------------------------------------------------------------------


class TestStageTransitions:
    """Tests for StageTransitions dataclass."""

    def test_defaults(self):
        """Default fields are empty."""
        st = StageTransitions(stage_name="test")
        assert st.targets == set()
        assert st.dynamic_returns == []

    def test_targets_include_none(self):
        """None is a valid target (terminal)."""
        st = StageTransitions(stage_name="test", targets={None, "next"})
        assert None in st.targets
        assert "next" in st.targets


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for validation."""

    def test_single_terminal_stage(self):
        """Workflow with a single stage that returns None is valid."""
        stage_map: dict[str, Stage[Any]] = {
            "only": TerminalStage(),
        }
        result = validate_workflow(stage_map, "only")
        # TerminalStage.name is "terminal" but we're using it with key "only"
        # The stage_map key is what matters for validation
        # However _extract_returns_from_run reads the actual source, so it will
        # correctly detect return None
        assert result.is_valid

    def test_empty_stage_map_with_missing_initial(self):
        """Empty stage map produces error about missing initial stage."""
        result = validate_workflow({}, "start")
        assert not result.is_valid
        assert any("not in stage map" in e for e in result.errors)

    def test_multiple_invalid_transitions(self):
        """Multiple stages with invalid transitions all produce errors."""

        class BadA(Stage[Any]):
            name = "bad_a"

            async def run(self) -> str | None:
                return "ghost_1"

        class BadB(Stage[Any]):
            name = "bad_b"

            async def run(self) -> str | None:
                return "ghost_2"

        stage_map: dict[str, Stage[Any]] = {
            "bad_a": BadA(),
            "bad_b": BadB(),
        }
        result = validate_workflow(stage_map, "bad_a")
        # Both bad targets should be reported
        target_errors = [e for e in result.errors if "not in the stage map" in e]
        assert len(target_errors) >= 2

    def test_ternary_with_none_branch(self):
        """Ternary that returns None in one branch is detected as terminal."""

        class TernaryNoneStage(Stage[Any]):
            name = "ternary_none"

            async def run(self) -> str | None:
                return "next" if False else None

        stage_map: dict[str, Stage[Any]] = {
            "ternary_none": TernaryNoneStage(),
            "next": TerminalStage(),
        }
        result = validate_workflow(stage_map, "ternary_none")
        # Should find None in targets, so it can terminate
        assert result.is_valid
