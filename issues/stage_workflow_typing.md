# Stage Workflow Typing

## Status: OPEN

## Problem

Stages need to access workflow-specific methods but receive only `node` parameter. This requires casting:

```python
# vote_train_set_stage.py
await cast("BasicLearningWorkflowModel", node.get_learning_workflow()).vote(...)
```

This is the same problem we solved for Commands with `Command[W]` generic pattern.

## Current Workaround

Using `cast()` everywhere stages access workflow methods.

## Proposed Solution

Apply similar pattern to stages:

```python
from typing import TypeVar, Generic

W = TypeVar("W", bound="LearningWorkflowModel")

class Stage(Generic[W]):
    @staticmethod
    async def execute(workflow: W, node: Node, state: LocalNodeState, ...) -> None:
        ...

# Usage
class VoteTrainSetStage(Stage[BasicLearningWorkflowModel]):
    @staticmethod
    async def execute(workflow: BasicLearningWorkflowModel, node: Node, ...) -> None:
        await workflow.vote(...)  # Type-safe!
```

## Alternative: Pass workflow directly

Instead of passing `node` and calling `node.get_learning_workflow()`, pass the typed workflow:

```python
# In workflow model
await VoteTrainSetStage.execute(
    workflow=self,  # Already typed as BasicLearningWorkflowModel
    node=self.node,
    state=self.local_state,
)
```

## Affected Files

- `p2pfl/stages/base_node/vote_train_set_stage.py`
- `p2pfl/stages/base_node/gossip_full_model_stage.py`
- `p2pfl/stages/base_node/gossip_partial_model_stage.py`
- Other stages that access workflow methods

## Related Issues

- `issues/solved/command_workflow_typing.md` - Similar solution for Commands
- `issues/stage_inheritance_design.md` - Other stage design issues
