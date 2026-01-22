# NodeWorkflowModel State Machine Overhead

## Problem

`NodeWorkflowModel` uses a pytransitions state machine for simple linear lifecycle management. This adds unnecessary complexity for operations that don't benefit from state machine patterns.

## Current Architecture

```
Node
  └── NodeWorkflowModel (state machine)  ← OVERHEAD
        └── LearningWorkflowModel (state machine)  ← NEEDED
              ├── BasicLearningWorkflowModel
              └── AsyncLearningWorkflowModel
```

## What NodeWorkflowModel Does

1. `start()` - initialize resources
2. `connect()/disconnect()` - peer management
3. `start_learning()` - delegates to LearningWorkflow
4. `stop()` - cleanup

This is a **linear lifecycle**, not a complex state machine:

```
idle → running → [learning] → stopped
```

## Why LearningWorkflow Needs a State Machine

- Multiple phases: setup, training, aggregation, evaluation
- Event-driven transitions (model received, votes collected, etc.)
- Different workflows (BasicDFL vs AsyncDFL) with different state graphs
- Timeouts, retries, error handling per state

## Why NodeWorkflow Does NOT Need a State Machine

- Always the same linear flow
- No event-driven branching
- No complex state-dependent behavior
- Operations are simple async methods

## Current Overhead

- `TimeoutMachine` instantiation for trivial states
- `WorkflowMachineManager` singleton to coordinate machines
- `get_states()` and `get_transitions()` boilerplate
- pytransitions callbacks for simple operations
- Harder to understand and debug

## Proposed Simplification

Replace `NodeWorkflowModel` state machine with a simple class:

```python
class NodeLifecycle:
    def __init__(self, node: Node):
        self.node = node
        self.is_running: bool = False
        self.learning_workflow: LearningWorkflowModel | None = None

    async def start(self) -> None:
        """Start the node."""
        if self.is_running:
            raise NodeRunningException()
        # Initialize resources
        self.is_running = True

    async def stop(self) -> None:
        """Stop the node."""
        if self.learning_workflow:
            await self.learning_workflow.stop_learning()
        # Cleanup resources
        self.is_running = False

    async def start_learning(self, workflow_type: WorkflowType, **kwargs) -> None:
        """Start the learning workflow."""
        if not self.is_running:
            raise NodeNotRunningException()
        # Build and start the appropriate LearningWorkflowModel
        self.learning_workflow = build_workflow(workflow_type, self.node, **kwargs)
        await self.learning_workflow.setup(**kwargs)
```

## Benefits of Simplification

1. **Easier to understand** - just async methods, no state machine concepts
2. **Less code** - remove states, transitions, callbacks boilerplate
3. **Faster** - no pytransitions overhead for simple operations
4. **Clearer separation** - Node lifecycle vs Learning workflow
5. **Easier testing** - simple methods vs state machine assertions

## Affected Files

- `p2pfl/stages/workflows/node_workflow.py`
- `p2pfl/node.py`
- `p2pfl/stages/workflows/workflow_state_manager.py`
- `p2pfl/stages/workflows/builder/workflow_director.py`

## Migration Path

1. Extract learning workflow creation from `NodeWorkflowModel`
2. Replace state machine methods with simple async methods
3. Keep `WorkflowMachineManager` only for `LearningWorkflowModel`
4. Update `Node` to use new simplified lifecycle
