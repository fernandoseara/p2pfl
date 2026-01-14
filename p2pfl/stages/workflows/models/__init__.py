"""Workflow models module."""

from p2pfl.stages.workflows.models.basicDFL.basic_learning_workflow import BasicLearningWorkflowModel
from p2pfl.stages.workflows.models.learning_workflow_model import LearningWorkflowModel
from p2pfl.stages.workflows.models.workflow_model import WorkflowModel

__all__ = [
    "BasicLearningWorkflowModel",
    "LearningWorkflowModel",
    "WorkflowModel",
]
