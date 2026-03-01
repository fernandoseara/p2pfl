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

"""Tests for the workflow factory."""

import pytest

from p2pfl.workflow.async_dfl.workflow import AsyncDFL
from p2pfl.workflow.basic_dfl.workflow import BasicDFL
from p2pfl.workflow.factory import create_workflow, list_workflows, register_workflow


class TestWorkflowFactory:
    """Tests for create_workflow factory function."""

    def test_create_basic_workflow(self):
        """Test creating a basic workflow."""
        result = create_workflow("basic")
        assert isinstance(result, BasicDFL)

    def test_create_async_workflow(self):
        """Test creating an async workflow."""
        result = create_workflow("async")
        assert isinstance(result, AsyncDFL)

    def test_create_unknown_workflow_raises(self):
        """Test that unknown workflow name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown workflow"):
            create_workflow("invalid_type")

    def test_list_workflows(self):
        """Test listing registered workflows."""
        names = list_workflows()
        assert "basic" in names
        assert "async" in names

    def test_register_duplicate_raises(self):
        """Test that registering a duplicate name raises ValueError."""
        with pytest.raises(ValueError, match="already registered"):
            register_workflow("basic", BasicDFL)
