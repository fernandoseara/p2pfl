# Stage Inheritance Design Issue

## Problem

Some stages in `asyDFL/` are being used as classical functions rather than as proper workflow stages.

### Expected Stage Pattern (BasicDFL)

Stages should:
- Have a uniform signature: `execute(...) -> None`
- Store results in state objects (network_state, local_state)
- Be interchangeable through the base `Stage` interface

### Actual Usage (AsyDFL)

These stages are called as regular functions with return values:

```python
# async_learning_workflow.py:399-408
neighbor_priorities = await ComputePriorityStage.execute(
    network_state=self.network_state,
    node=self.node,
)

self.candidates = await SelectNeighborsStage.execute(
    neighbor_priorities=neighbor_priorities,
    node=self.node,
)
```

This breaks the stage abstraction:
- `ComputePriorityStage.execute()` returns `list[tuple[str, float]]`
- `SelectNeighborsStage.execute()` returns `list[str]`
- They have specific typed parameters, not `*args, **kwargs`

## Why This Is a Problem

1. **No polymorphism**: Can't iterate over stages or use them interchangeably
2. **Violates LSP**: Subclass signatures incompatible with base class
3. **Mixed paradigms**: Some stages follow the pattern, others are just functions with a class wrapper

## Status: MUST FIX

Cannot use `# type: ignore[override]` - need proper solution.

## Solution Options

Refactor `ComputePriorityStage` and `SelectNeighborsStage` to follow the standard pattern:
- Store results in `AsyncNetworkState` instead of returning them
- Use uniform `execute(network_state, node) -> None` signature
- Or: Remove `Stage` inheritance entirely if they're meant to be utility functions

## Affected Files

- `p2pfl/stages/asyDFL/compute_priority_stage.py`
- `p2pfl/stages/asyDFL/select_neighbor_stage.py`
- `p2pfl/stages/workflows/models/asyncFL/async_learning_workflow.py`
