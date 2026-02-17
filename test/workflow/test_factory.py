#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
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

from unittest.mock import MagicMock, patch

import pytest

from p2pfl.workflow.async_dfl.workflow import AsyncDFL
from p2pfl.workflow.basic_dfl.workflow import BasicDFL
from p2pfl.workflow.factory import WorkflowType, create_workflow


@pytest.fixture
def mock_node():
    """Create a mock node for testing."""
    node = MagicMock()
    node.address = "test_node_address"
    node.get_communication_protocol.return_value = MagicMock()
    node.get_learner.return_value = MagicMock()
    node.get_learner.return_value.get_model.return_value = MagicMock()
    return node


class TestWorkflowFactory:
    """Tests for create_workflow factory function."""

    def test_create_basic_workflow(self, mock_node):
        """Test creating a basic workflow."""
        result = create_workflow(WorkflowType.BASIC, mock_node)

        assert isinstance(result, BasicDFL)
        assert result.node == mock_node

    def test_create_async_workflow(self, mock_node):
        """Test creating an async workflow."""
        # Patch CustomModelFactory to avoid needing a real TensorFlow model
        with patch("p2pfl.workflow.async_dfl.workflow.CustomModelFactory") as mock_factory:
            mock_factory.create_model.return_value = MagicMock()
            result = create_workflow(WorkflowType.ASYNC, mock_node)

            assert isinstance(result, AsyncDFL)
            assert result.node == mock_node

    def test_create_unknown_workflow_raises(self, mock_node):
        """Test that unknown workflow type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown workflow type"):
            create_workflow("invalid_type", mock_node)  # type: ignore

    def test_workflow_types_enum(self):
        """Test WorkflowType enum values."""
        assert WorkflowType.BASIC.value == "basic"
        assert WorkflowType.ASYNC.value == "async"
        assert len(WorkflowType) == 2
