# Command[W] Generic Type Pattern

## Status: NOT IMPLEMENTED - Circular Import Issue

The `Command[W]` generic pattern was designed but **never committed** due to circular import issues that couldn't be cleanly resolved.

## The Idea

Allow commands to declare which workflow type they need for type-safe access:

```python
# The idea was to do this:
class VoteTrainSetCommand(Command[BasicLearningWorkflowModel]):
    async def execute(self, source: str, round: int, *args, **kwargs) -> None:
        await self.workflow.vote(source, round, votes)  # Type-safe!
```

## Why It Doesn't Work

Creates an unresolvable circular import:

```
Workflow → imports Command (to call .get_name())
Command → imports Workflow (for Command[W] type parameter)
         ↓
    CIRCULAR IMPORT
```

### Attempted Solutions

1. **TYPE_CHECKING for workflow imports** - Only breaks one direction; workflow still imports commands
2. **String literals for type parameters** - Still requires import somewhere for runtime
3. **Lazy imports** - Ugly and error-prone

None of these solutions were clean enough to justify the complexity.

## Current Implementation

Commands use a simple base class without generics:

```python
class Command(abc.ABC):
    """Base class for all commands."""

    @property
    def workflow(self) -> Any:
        """Get the workflow (untyped)."""
        return self.node.get_learning_workflow()
```

Commands just use `self.workflow` without type safety:

```python
class VoteTrainSetCommand(Command):
    async def execute(self, source: str, round: int, *args, **kwargs) -> None:
        await self.workflow.vote(source, round, votes)  # Works, but no type checking
```

## Trade-offs

| Aspect | Generic Pattern | Current (Simple) |
|--------|-----------------|------------------|
| Type safety | Full static typing | Runtime only |
| IDE autocomplete | Yes | No (Any type) |
| Circular imports | Broken | Works |
| Complexity | Higher | Lower |

## Future Consideration

If type safety becomes important, consider:

1. **Protocol-based typing**: Define protocols for workflow capabilities instead of concrete types
2. **Command name constants**: Move `get_name()` to a shared module to break workflow→command dependency
3. **Restructure the architecture**: Workflows shouldn't need to import commands directly
