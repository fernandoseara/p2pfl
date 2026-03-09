# Hierarchical Federated Learning (HFL) - Design Document

**Status:** Proposed
**Date:** 2026-02-24
**Updated:** 2026-03-05 (Updated for asyncio_support branch architecture)

---

## 1. Motivation

The current p2pfl system implements a fully decentralized (flat) peer-to-peer federated learning workflow where every node is equal: all nodes train, gossip partial models, and aggregate to reach global consensus. While this works well for small-to-medium networks, it has limitations in larger deployments:

- **Communication overhead:** Every node must exchange models with every other node in the training set via gossip, leading to O(n^2) communication cost.
- **Scalability:** As the number of nodes grows, the gossip convergence time increases significantly.
- **Network topology mismatch:** In real-world deployments (e.g., IoT, mobile edge computing), nodes are naturally organized in clusters with a local gateway or edge server. The flat P2P topology does not reflect this physical structure.

**Hierarchical Federated Learning (HFL)** introduces a two-level hierarchy that groups worker nodes under edge nodes. Workers train and send models to their local edge, edge nodes aggregate locally and then perform inter-edge gossip to reach global consensus, and finally distribute the global model back to workers. This reduces cross-cluster communication and matches real-world network topologies.

---

## 2. Design Decisions

The following decisions were made upfront to scope the design:

| Decision | Choice | Rationale |
|---|---|---|
| **Role assignment** | `role` field in `HFLContext` | Roles are configured at the workflow level via a typed context (`HFLContext`), keeping the `Node` class completely untouched. The workflow carries all HFL-specific state in its specialized context. |
| **Edge training** | Train + Aggregate | Edge nodes train on their own local data AND aggregate worker models. They participate in both levels of the hierarchy. |
| **Inter-edge communication** | P2P gossip with dedicated handlers | Edge nodes use async message handlers decorated with `@on_message`, scoped to specific stages via the `during` parameter. No central coordinator. |
| **Hierarchy depth** | 2-level only | Workers -> Edge nodes. Simpler design that covers the primary use case without recursive complexity. |

---

## 3. Architecture Overview

### 3.1 Current Workflow (Flat P2P)

In the existing system, all nodes follow the same workflow:

```
SetupStage
    |
    v
RoundInitStage
    |
    v
VotingStage
    |
    v
LearningStage    (evaluate, train, gossip partial models, aggregate)
    |
    v
RoundInitStage   (loop back or finish)
    |
    v
FinishStage
```

Reference: `p2pfl/workflow/basic_dfl/workflow.py`, `p2pfl/workflow/basic_dfl/stages/`

### 3.2 Proposed HFL Workflow

In HFL, each role follows a different stage pipeline within the same workflow. The initial `HFLSetupStage` branches into role-specific paths based on `ctx.role` by conditionally returning different stage names from its `async def run()` method:

**Worker pipeline:**
```
setup (HFLSetupStage - branches by role)
   |
   v [if ctx.role == "worker", return "worker_train"]
worker_train (HFLWorkerTrainStage)
   |  (train locally, send model to edge via cp.send())
   v
worker_wait_global (HFLWorkerWaitGlobalStage)
   |  (await asyncio.Event, receives @on_message("global_model"))
   v
round_finished (HFLRoundFinishedStage)
   |  (loop back to "worker_train" or return None)
   v
```

**Edge pipeline:**
```
setup (HFLSetupStage - branches by role)
   |
   v [if ctx.role == "edge", return "edge_local_train"]
edge_local_train (HFLEdgeLocalTrainStage)
   |  (train on own data, prepare aggregator for workers)
   v
edge_aggregate_workers (HFLEdgeAggregateWorkersStage)
   |  (receives @on_message("worker_model"), aggregates)
   v
edge_gossip (HFLEdgeGossipStage)
   |  (gossip with @on_message("edge_model", during={"edge_gossip"}))
   v
edge_distribute (HFLEdgeDistributeStage)
   |  (push global model to all workers)
   v
round_finished (HFLRoundFinishedStage)
   |  (loop back to "edge_local_train" or return None)
   v
```

### 3.3 Per-Round Data Flow

```
WORKER                                  EDGE                                  EDGE (peer)
------                                  ----                                  -----------
worker_train (HFLWorkerTrainStage)      edge_local_train (HFLEdgeLocalTrainStage)
  |-- await ctx.learner.fit()             |-- await ctx.learner.fit()
  |-- send model to edge --------+        |-- store own model in ctx.peers[self].model
  |   (via await ctx.cp.send())   |       |
  v                               |       |
worker_wait_global                |       v
  |                               +-----> edge_aggregate_workers (HFLEdgeAggregateWorkersStage)
  |                                       |-- @on_message("worker_model") stores in ctx.peers[src].model
  |-- ctx.global_model_ready.clear()      |-- await wait_with_timeout(self._workers_complete, ...)
  |-- await ctx.global_model_ready.wait() |-- models = [p.model for p in ctx.peers.values() if p.model]
  |   (asyncio.Event)                     |-- agg = ctx.aggregator.aggregate(models)  (STATELESS)
  |                                       |-- ctx.learner.set_model(agg)
  |                                       |-- *** FLATTEN contributors to [ctx.address] ***
  |                                       |
  |                                       v
  |                                     edge_gossip (HFLEdgeGossipStage)
  |                                       |-- reset ctx.peers for edge phase
  |                                       |-- store own model in ctx.peers[self].model (contributors=[self])
  |                                       |-- @on_message("edge_model", during={"edge_gossip"})
  |                                       |     stores in ctx.peers[src].model
  |                                       |-- await wait_with_timeout(self._edges_complete, ...)
  |                                       |-- models = [p.model for p in ctx.peers.values() if p.model]
  |                                       |-- global_model = ctx.aggregator.aggregate(models)  (STATELESS)
  |                                       |-- ctx.learner.set_model(global_model)
  |                                       |
  |                                       v
  |                                     edge_distribute (HFLEdgeDistributeStage)
  |    <--- global model ------------------|-- for w in ctx.worker_addrs: await ctx.cp.send(w, model)
  |    (via @on_message("global_model"))  |   (sends "global_model" message)
  v                                       v
round_finished (HFLRoundFinishedStage)  round_finished (HFLRoundFinishedStage)
  |-- reset ctx.peers                     |-- reset ctx.peers
  |-- ctx.experiment.increase_round()     |-- ctx.experiment.increase_round()
  |-- loop (return "worker_train")        |-- loop (return "edge_local_train")
  |   or finish (return None)             |   or finish (return None)
```

---

## 4. Detailed Implementation Plan

### 4.1 Create `HFLContext` Dataclass

**New file:** `p2pfl/workflow/hfl/context.py`

All HFL-specific state lives in an `HFLContext` dataclass that extends `WorkflowContext`. Neither `Node` nor `NodeState` are modified.

```python
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from p2pfl.workflow.engine.context import WorkflowContext

if TYPE_CHECKING:
    from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


@dataclass
class HFLPeerState:
    """Per-peer mutable state for HFL workflow."""

    round_number: int = 0
    model: P2PFLModel | None = None
    aggregated_from: list[str] = field(default_factory=list)

    def reset_round(self) -> None:
        """Reset per-round mutable state."""
        self.model = None
        self.aggregated_from.clear()


@dataclass
class HFLContext(WorkflowContext):
    """
    Typed context for Hierarchical Federated Learning.

    Extends WorkflowContext with role-specific state for two-level
    hierarchical topology (workers -> edges).
    """

    # Role configuration
    role: str  # "worker" or "edge"

    # Topology (configured by TopologyFactory.connect_hierarchical)
    edge_addr: str | None = None  # For workers: assigned edge
    worker_addrs: list[str] = field(default_factory=list)  # For edges: managed workers
    edge_peers: list[str] = field(default_factory=list)  # For edges: other edge nodes

    # Per-peer tracking (similar to BasicDFLContext.peers)
    peers: dict[str, HFLPeerState] = field(default_factory=dict)

    # Synchronization (asyncio, not threading)
    global_model_ready: asyncio.Event = field(default_factory=asyncio.Event)
```

**Key changes from the old architecture:**
- Inherits from `WorkflowContext` (includes `address`, `learner`, `aggregator`, `cp`, `generator`, `experiment`)
- Uses `asyncio.Event` instead of `threading.Event` for worker synchronization
- Follows the `peers` dict pattern from `BasicDFLContext` for per-peer tracking
- No locks needed (asyncio is single-threaded cooperative)
- Removed separate `edge_models_aggregated` dict (use `peers` instead)

**Rationale:** By keeping all HFL state in a typed context:
- `Node` remains completely unchanged
- The flat P2P workflows (`BasicDFL`, `AsyncDFL`) are entirely unaffected
- Context is created by `HFLWorkflow.create_context()` and passed to all stages as `self.ctx`
- Topology helpers configure the context fields directly via `node.workflow.ctx`

**Reference:** `p2pfl/workflow/basic_dfl/context.py`

### 4.2 Create `HFL` Workflow

**New file:** `p2pfl/workflow/hfl/workflow.py`

The `HFL` workflow class extends `Workflow[HFLContext]` to implement hierarchical federated learning:

```python
from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from p2pfl.workflow.engine.experiment import Experiment
from p2pfl.workflow.engine.stage import Stage
from p2pfl.workflow.engine.workflow import Workflow
from p2pfl.workflow.hfl.context import HFLContext
from p2pfl.workflow.hfl.stages import (
    HFLEdgeAggregateWorkersStage,
    HFLEdgeDistributeStage,
    HFLEdgeGossipStage,
    HFLEdgeLocalTrainStage,
    HFLRoundFinishedStage,
    HFLSetupStage,
    HFLWorkerTrainStage,
    HFLWorkerWaitGlobalStage,
)

if TYPE_CHECKING:
    from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
    from p2pfl.learning.aggregators.aggregator import Aggregator
    from p2pfl.learning.frameworks.learner import Learner


class HFL(Workflow[HFLContext]):
    """
    Hierarchical Federated Learning workflow.

    Two-level hierarchy: workers train locally and send models to their
    assigned edge node. Edge nodes aggregate worker models, perform
    inter-edge gossip for global consensus, then distribute the global
    model back to workers.

    Flow (worker): setup -> worker_train -> worker_wait_global -> round_finished -> loop
    Flow (edge):   setup -> edge_local_train -> edge_aggregate_workers ->
                   edge_gossip -> edge_distribute -> round_finished -> loop
    """

    initial_stage = "setup"
    context_class = HFLContext

    def get_stages(self) -> list[Stage[HFLContext]]:
        """Return all HFL stages (both worker and edge paths)."""
        return [
            HFLSetupStage(),
            # Worker stages
            HFLWorkerTrainStage(),
            HFLWorkerWaitGlobalStage(),
            # Edge stages
            HFLEdgeLocalTrainStage(),
            HFLEdgeAggregateWorkersStage(),
            HFLEdgeGossipStage(),
            HFLEdgeDistributeStage(),
            # Shared
            HFLRoundFinishedStage(),
        ]

    def create_context(
        self,
        address: str,
        learner: Learner,
        aggregator: Aggregator,
        cp: CommunicationProtocol,
        generator: random.Random,
        experiment: Experiment,
        role: str,  # Required HFL-specific kwarg
        edge_addr: str | None = None,
        worker_addrs: list[str] | None = None,
        edge_peers: list[str] | None = None,
        **kwargs: Any,
    ) -> HFLContext:
        """
        Build HFL context with role and topology configuration.

        Args:
            role: "worker" or "edge"
            edge_addr: (workers only) Address of assigned edge node
            worker_addrs: (edges only) List of worker addresses to manage
            edge_peers: (edges only) List of other edge node addresses
        """
        return HFLContext(
            address=address,
            learner=learner,
            aggregator=aggregator,
            cp=cp,
            generator=generator,
            experiment=experiment,
            role=role,
            edge_addr=edge_addr,
            worker_addrs=worker_addrs or [],
            edge_peers=edge_peers or [],
        )
```

**Key changes from the old architecture:**
- No manual command registration (handled automatically via `@on_message` decorators in stages)
- Typed as `Workflow[HFLContext]` with generic context
- `initial_stage` and `get_stages()` instead of constructor with `StageFactory`
- `create_context()` method accepts HFL-specific kwargs (role, topology info)
- No `run()` override needed (base class handles everything)
- All stages returned from `get_stages()` (branching happens in each stage's `run()` method)

**Reference:** `p2pfl/workflow/basic_dfl/workflow.py`

### 4.3 Message Handlers

In the new architecture, there are **no dedicated Command classes**. Instead, message handlers are declared directly in stages using the `@on_message` decorator.

The generic `WorkflowCommand` (in `p2pfl/communication/commands/workflow/workflow_command.py`) automatically routes messages to the appropriate handler based on:

1. **Message name** (e.g., `"worker_model"`, `"global_model"`)
2. **Current stage** (via `during` parameter in decorator)
3. **Payload type** (regular args vs weights)

#### Handler Registration

Handlers are automatically discovered by the workflow engine during `_compose()`:
1. Scans all stages for `@on_message` decorated methods
2. Builds `_handlers` map: `{message_name: [(callback, entry), ...]}`
3. Node creates `WorkflowCommand` instances for each message
4. At runtime, `WorkflowCommand.execute()` looks up handlers for current stage

#### Example Handler 1: Worker Model (in `HFLEdgeAggregateWorkersStage`)

```python
from p2pfl.workflow.engine.message import on_message
from p2pfl.workflow.engine.stage import Stage
from p2pfl.workflow.hfl.context import HFLContext


class HFLEdgeAggregateWorkersStage(Stage[HFLContext]):
    """Edge aggregates worker models."""

    @on_message("worker_model", weights=True)
    async def handle_worker_model(
        self,
        source: str,
        round: int,
        weights: bytes,
        contributors: list[str] | None,
        num_samples: int | None,
    ) -> None:
        """Receive trained model from worker."""
        ctx = self.ctx
        if round != ctx.experiment.round:
            return

        model = ctx.learner.get_model().build_copy(
            params=weights,
            num_samples=num_samples,
            contributors=list(contributors or []),
        )
        ctx.aggregator.add_model(model)
```

#### Example Handler 2: Global Model (in `HFLWorkerWaitGlobalStage`)

```python
class HFLWorkerWaitGlobalStage(Stage[HFLContext]):
    """Worker waits for global model from edge."""

    @on_message("global_model", weights=True)
    async def handle_global_model(
        self,
        source: str,
        round: int,
        weights: bytes,
        contributors: list[str] | None,
        num_samples: int | None,
    ) -> None:
        """Receive global model from edge."""
        ctx = self.ctx
        model = ctx.learner.get_model().build_copy(params=weights, ...)
        ctx.learner.set_model(model)
        ctx.global_model_ready.set()  # Asyncio event
```

#### Example Handler 3: Edge Model (in `HFLEdgeGossipStage`)

```python
class HFLEdgeGossipStage(Stage[HFLContext]):
    """Inter-edge gossip for global aggregation."""

    @on_message("edge_model", weights=True, during={"edge_gossip"})
    async def handle_edge_model(
        self,
        source: str,
        round: int,
        weights: bytes,
        contributors: list[str] | None,
        num_samples: int | None,
    ) -> None:
        """Receive edge model during inter-edge gossip."""
        ctx = self.ctx
        model = ctx.learner.get_model().build_copy(params=weights, ...)
        ctx.aggregator.add_model(model)

        # Track in peers dict instead of separate dict
        if source not in ctx.peers:
            from p2pfl.workflow.hfl.context import HFLPeerState
            ctx.peers[source] = HFLPeerState()
        ctx.peers[source].aggregated_from = list(contributors or [])
```

**Key advantages of `@on_message` over dedicated Commands:**

1. **No manual registration:** Handlers are automatically discovered and registered by the workflow engine
2. **Type-safe:** Handlers are instance methods with access to `self.ctx` (typed context)
3. **Scoped activation:** Use `during={"stage_name"}` to activate handlers only in specific stages
4. **Cleaner code:** No need for separate command files - logic lives with the stage that uses it
5. **No state pollution:** Each stage's handlers are naturally isolated

**Reference:** `p2pfl/communication/commands/workflow/workflow_command.py`

### 4.4 Create HFL Stages

**New directory:** `p2pfl/workflow/hfl/stages/`

All stages follow the new pattern:
- Inherit from `Stage[HFLContext]`
- Implement `async def run(self) -> str | None`
- Access context via `self.ctx` (typed as `HFLContext`)
- Return next stage name (string) or `None` to finish
- Can define `@on_message` handlers scoped to the stage

**General pattern:**

```python
from p2pfl.workflow.engine.stage import Stage
from p2pfl.workflow.hfl.context import HFLContext


class MyCustomStage(Stage[HFLContext]):
    """Docstring."""

    name = "custom"  # Optional: override auto-derived "my_custom"

    async def run(self) -> str | None:
        """Execute stage logic."""
        ctx = self.ctx  # Access typed context

        # Stage logic here

        return "next_stage_name"  # Or None to finish

    # Optional: message handlers scoped to this stage
    @on_message("some_message", weights=True)
    async def handle_message(self, source: str, round: int, ...) -> None:
        """Handle message."""
        pass
```

#### 4.4.1 HFLSetupStage

**New file:** `p2pfl/workflow/hfl/stages/setup.py`

Initializes the workflow and branches by role:

```python
class HFLSetupStage(Stage[HFLContext]):
    """Setup and initial synchronization, then branch by role."""

    name = "setup"  # Override auto-derived "hfl_setup"

    async def run(self) -> str | None:
        """
        Initialize peers, wait for synchronization, then branch.

        Returns:
            "worker_train" if worker, "edge_local_train" if edge.
        """
        ctx = self.ctx

        # Initialize experiment
        ctx.experiment.start()
        logger.info(ctx.address, "⏳ Starting HFL training.")

        # Wait for topology to be ready (optional)
        # ... synchronization logic ...

        # Branch by role
        if ctx.role == "worker":
            return "worker_train"
        elif ctx.role == "edge":
            return "edge_local_train"
        else:
            raise ValueError(f"Unknown role: {ctx.role}")
```

#### 4.4.2 HFLWorkerTrainStage

**New file:** `p2pfl/workflow/hfl/stages/worker_train.py`

Worker trains locally, then sends model to assigned edge:

```python
class HFLWorkerTrainStage(Stage[HFLContext]):
    """Worker trains and sends model to edge."""

    name = "worker_train"

    async def run(self) -> str | None:
        """Train locally and send model to edge."""
        ctx = self.ctx

        # Train
        await ctx.learner.fit()

        # Encode and send model to edge
        model = ctx.learner.get_model()
        model.set_contribution([ctx.address], model.get_num_samples())
        encoded = model.encode_parameters()

        msg = ctx.cp.build_weights(
            "worker_model",
            ctx.experiment.round,
            encoded,
            contributors=[ctx.address],
            num_samples=model.get_num_samples(),
        )
        await ctx.cp.send(ctx.edge_addr, msg)

        return "worker_wait_global"
```

#### 4.4.3 HFLWorkerWaitGlobalStage

**New file:** `p2pfl/workflow/hfl/stages/worker_wait_global.py`

Worker waits for global model from edge. Has `@on_message("global_model")` handler that receives the model and sets the event:

```python
class HFLWorkerWaitGlobalStage(Stage[HFLContext]):
    """Worker waits for global model from edge."""

    name = "worker_wait_global"

    @on_message("global_model", weights=True)
    async def handle_global_model(
        self,
        source: str,
        round: int,
        weights: bytes,
        contributors: list[str] | None,
        num_samples: int | None,
    ) -> None:
        """Receive global model from edge."""
        ctx = self.ctx
        model = ctx.learner.get_model().build_copy(params=weights, ...)
        ctx.learner.set_model(model)
        ctx.global_model_ready.set()

    async def run(self) -> str | None:
        """Wait for global model."""
        ctx = self.ctx
        ctx.global_model_ready.clear()

        try:
            await asyncio.wait_for(
                ctx.global_model_ready.wait(),
                timeout=Settings.training.AGGREGATION_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(ctx.address, "Timeout waiting for global model")

        return "round_finished"
```

#### 4.4.4 HFLEdgeLocalTrainStage

**New file:** `p2pfl/workflow/hfl/stages/edge_local_train.py`

Edge trains on its own local data and stores its model in `ctx.peers` for the upcoming worker aggregation phase.

```python
class HFLEdgeLocalTrainStage(Stage[HFLContext]):
    """Edge trains locally and stores its own model."""

    name = "edge_local_train"

    async def run(self) -> str | None:
        ctx = self.ctx
        address = ctx.address

        # Evaluate (optional)
        await evaluate_and_broadcast(ctx)

        # Train
        logger.info(address, "Edge training...")
        await ctx.learner.fit()
        logger.info(address, "Edge training done.")

        # Store own model in peers dict for aggregation
        peer = ctx.peers.get(address)
        if peer is None:
            ctx.peers[address] = HFLPeerState()
            peer = ctx.peers[address]
        peer.model = ctx.learner.get_model()

        return "edge_aggregate_workers"
```

#### 4.4.5 HFLEdgeAggregateWorkersStage

**New file:** `p2pfl/workflow/hfl/stages/edge_aggregate_workers.py`

Edge waits for all worker models to arrive via `@on_message("worker_model")`, then aggregates them using the **stateless** `aggregator.aggregate()` call. Models are collected in `ctx.peers` and completion is tracked with an `asyncio.Event`.

```python
class HFLEdgeAggregateWorkersStage(Stage[HFLContext]):
    """Edge aggregates worker models + own model."""

    name = "edge_aggregate_workers"

    def __init__(self) -> None:
        super().__init__()
        self._workers_complete = asyncio.Event()

    def _all_worker_models_received(self, ctx: HFLContext) -> bool:
        expected = set(ctx.worker_addrs) | {ctx.address}
        received = {addr for addr, p in ctx.peers.items() if p.model is not None}
        return expected.issubset(received)

    @on_message("worker_model", weights=True)
    async def handle_worker_model(
        self,
        source: str,
        round: int,
        weights: bytes,
        contributors: list[str] | None,
        num_samples: int | None,
    ) -> None:
        """Receive trained model from a worker."""
        ctx = self.ctx
        if round != ctx.experiment.round:
            return
        if contributors is None or num_samples is None:
            raise ValueError("Contributors and num_samples are required")

        model = ctx.learner.get_model().build_copy(
            params=weights,
            num_samples=num_samples,
            contributors=list(contributors),
        )
        if source not in ctx.peers:
            ctx.peers[source] = HFLPeerState()
        ctx.peers[source].model = model

        if self._all_worker_models_received(ctx):
            self._workers_complete.set()

    async def run(self) -> str | None:
        ctx = self.ctx
        self._workers_complete.clear()

        # Wait for all worker models
        if not self._all_worker_models_received(ctx):
            await wait_with_timeout(
                self._workers_complete,
                Settings.training.AGGREGATION_TIMEOUT,
                ctx.address,
                "Timeout waiting for worker models.",
            )

        # Aggregate all models (stateless call)
        models = [p.model for p in ctx.peers.values() if p.model is not None]
        agg_model = ctx.aggregator.aggregate(models)
        ctx.learner.set_model(agg_model)

        # *** CRITICAL: Flatten contributors for inter-edge phase ***
        # The aggregated model has contributors=[edgeA, w1, w2, ...].
        # The inter-edge gossip tracks models by edge address only.
        # We must flatten to [ctx.address] so the edge-level peer
        # tracking correctly identifies this model as belonging to this edge.
        agg_model = ctx.learner.get_model()
        agg_model.set_contribution(
            contributors=[ctx.address],
            num_samples=agg_model.get_num_samples(),
        )

        return "edge_gossip"
```

**Why flatten contributors?** After local aggregation, the model has `contributors=[edgeA, w1, w2]`. The inter-edge gossip phase tracks models in `ctx.peers` keyed by edge address. When checking `_all_edge_models_received()`, only edge addresses are expected. If the model retains worker contributors, the gossip logic cannot correctly match models to their source edges. Flattening to `[ctx.address]` ensures clean edge-level identity for the gossip phase.

#### 4.4.6 HFLEdgeGossipStage

**New file:** `p2pfl/workflow/hfl/stages/edge_gossip.py`

Edge nodes share their locally-aggregated model with other edge nodes via `@on_message("edge_model")`, achieving global consensus. Models are collected in `ctx.peers` (reset for this phase) and aggregated with the stateless `aggregator.aggregate()`.

```python
class HFLEdgeGossipStage(Stage[HFLContext]):
    """Inter-edge gossip for global model consensus."""

    name = "edge_gossip"

    def __init__(self) -> None:
        super().__init__()
        self._edges_complete = asyncio.Event()

    def _all_edge_models_received(self, ctx: HFLContext) -> bool:
        expected = set(ctx.edge_peers) | {ctx.address}
        received = {addr for addr, p in ctx.peers.items() if p.model is not None}
        return expected.issubset(received)

    @on_message("edge_model", weights=True, during={"edge_gossip"})
    async def handle_edge_model(
        self,
        source: str,
        round: int,
        weights: bytes,
        contributors: list[str] | None,
        num_samples: int | None,
    ) -> None:
        """Receive edge model during inter-edge gossip."""
        ctx = self.ctx
        if round != ctx.experiment.round:
            return
        if contributors is None or num_samples is None:
            raise ValueError("Contributors and num_samples are required")

        model = ctx.learner.get_model().build_copy(
            params=weights,
            num_samples=num_samples,
            contributors=list(contributors),
        )
        if source not in ctx.peers:
            ctx.peers[source] = HFLPeerState()
        ctx.peers[source].model = model

        if self._all_edge_models_received(ctx):
            self._edges_complete.set()

    async def run(self) -> str | None:
        ctx = self.ctx
        self._edges_complete.clear()

        # Reset peers dict for edge-level aggregation phase
        for addr in list(ctx.peers.keys()):
            ctx.peers[addr].reset_round()

        # Store own model (already flattened, contributors=[ctx.address])
        ctx.peers[ctx.address] = HFLPeerState()
        ctx.peers[ctx.address].model = ctx.learner.get_model()

        # Send model to other edges
        model = ctx.learner.get_model()
        payload = ctx.cp.build_weights(
            "edge_model",
            ctx.experiment.round,
            model.encode_parameters(),
            model.get_contributors(),
            model.get_num_samples(),
        )
        for edge_addr in ctx.edge_peers:
            await ctx.cp.send(edge_addr, payload)

        # Wait for all edge models
        if not self._all_edge_models_received(ctx):
            await wait_with_timeout(
                self._edges_complete,
                Settings.training.GLOBAL_GOSSIP_TIMEOUT,
                ctx.address,
                "Timeout waiting for edge models.",
            )

        # Aggregate all edge models (stateless call)
        models = [p.model for p in ctx.peers.values() if p.model is not None]
        global_model = ctx.aggregator.aggregate(models)
        ctx.learner.set_model(global_model)

        return "edge_distribute"
```

**Note on gossip approach:** This simplified version uses direct send to all edge peers. For larger deployments with many edge nodes, the existing `gossip_weights` transport layer (which accepts generic callbacks) can be reused for partial gossip with the same pattern — just replace the direct sends with `ModelGate`-based gossiping as done in `LearningStage._gossip_partial_models()`.

#### 4.4.7 HFLEdgeDistributeStage

**New file:** `p2pfl/workflow/hfl/stages/edge_distribute.py`

Edge pushes the globally-aggregated model to all its workers via direct `await ctx.cp.send()`.

```python
class HFLEdgeDistributeStage(Stage[HFLContext]):
    """Edge distributes global model to workers."""

    name = "edge_distribute"

    async def run(self) -> str | None:
        ctx = self.ctx
        model = ctx.learner.get_model()
        payload = ctx.cp.build_weights(
            "global_model",
            ctx.experiment.round,
            model.encode_parameters(),
            model.get_contributors(),
            model.get_num_samples(),
        )
        for worker_addr in ctx.worker_addrs:
            await ctx.cp.send(worker_addr, payload)

        return "round_finished"
```

#### 4.4.8 HFLRoundFinishedStage

**New file:** `p2pfl/workflow/hfl/stages/round_finished.py`

Shared by both roles. Resets peer state, increments the round, and branches back by role or finishes.

```python
class HFLRoundFinishedStage(Stage[HFLContext]):
    """Round completion — shared by workers and edges."""

    name = "round_finished"

    async def run(self) -> str | None:
        ctx = self.ctx
        address = ctx.address

        # Reset peer state for next round
        for peer in ctx.peers.values():
            peer.reset_round()

        # Advance round
        ctx.experiment.increase_round(address)
        logger.info(address, f"Round {ctx.experiment.round} finished.")

        # Check if more rounds remain
        if ctx.experiment.round < ctx.experiment.total_rounds:
            if ctx.role == "worker":
                return "worker_train"
            else:
                return "edge_local_train"

        # Final evaluation
        await evaluate_and_broadcast(ctx)
        return None  # Workflow complete
```

### 4.5 Extend Topology Utilities

**File:** `p2pfl/utils/topologies.py`

Add `HIERARCHICAL` to the `TopologyType` enum and a helper method that configures context fields directly via `node.workflow.ctx`:

```python
class TopologyType(Enum):
    # ... existing types ...
    HIERARCHICAL = "hierarchical"
```

Add to `TopologyFactory`:

```python
@staticmethod
async def connect_hierarchical(
    edge_nodes: list[Node],
    worker_groups: list[list[Node]],
    edge_topology: TopologyType = TopologyType.FULL,
) -> None:
    """
    Set up a hierarchical FL topology.

    Establishes connections and configures each node's workflow context.

    Args:
        edge_nodes: List of edge nodes (must have HFL workflow assigned).
        worker_groups: worker_groups[i] contains the workers for edge_nodes[i].
        edge_topology: Topology among edge nodes (default: fully connected).
    """
    all_edge_addrs = [e.addr for e in edge_nodes]

    # 1. Connect each worker to its edge and configure contexts
    for edge, workers in zip(edge_nodes, worker_groups, strict=True):
        worker_addrs = [w.addr for w in workers]

        # Configure edge context (NOT hfl_context attribute)
        edge.workflow.ctx.worker_addrs = worker_addrs
        edge.workflow.ctx.edge_peers = [
            a for a in all_edge_addrs if a != edge.addr
        ]

        for worker in workers:
            # Connect worker <-> edge
            await worker.connect(edge.addr)
            await edge.connect(worker.addr)

            # Configure worker context
            worker.workflow.ctx.edge_addr = edge.addr

    # 2. Connect edge nodes to each other
    edge_matrix = TopologyFactory.generate_matrix(edge_topology, len(edge_nodes))
    await TopologyFactory.connect_nodes(edge_matrix, edge_nodes)
```

**Key change:** Contexts are configured via `node.workflow.ctx` (which is created by `create_context()` during `workflow.run()`), not via a separate `hfl_context` attribute.

This is a backward-compatible addition — `LearningWorkflow` and existing stages simply ignore the extra kwarg via `**kwargs`.

### 4.8 Add HFL-specific Settings (Optional)

**File:** `p2pfl/settings.py`

```python
@dataclass
class HFL:
    """Hierarchical Federated Learning settings."""
    WORKER_SEND_TIMEOUT: int = 120
    EDGE_AGGREGATION_TIMEOUT: int = 300
    GLOBAL_GOSSIP_TIMEOUT: int = 300
```

---

## 5. Key Design Rationale

### 5.1 Zero changes to Node

All HFL state lives in `HFLContext` (extends `WorkflowContext`), created by `HFL.create_context()`. The `Node` class is not modified at all. This means:
- The flat P2P workflows (`BasicDFL`, `AsyncDFL`) remain completely untouched
- Roles are a workflow-level concern, not a node-level concern
- A node's role can be changed by simply replacing its `workflow` with `create_workflow("hfl")`

### 5.2 Separate workflow, not conditionals in existing stages

New HFL stages live in `p2pfl/workflow/hfl/stages/` rather than adding if/else branches to existing stages. This keeps existing P2P workflows completely untouched. Using `BasicDFL` (the default) preserves current behavior exactly.

### 5.3 Reuse of existing Aggregator (stateless)

The `Aggregator` class (`p2pfl/learning/aggregators/aggregator.py`) is **completely stateless** — it exposes only `aggregate(models: list[P2PFLModel]) -> P2PFLModel`. There is no `set_nodes_to_aggregate()`, `add_model()`, `wait_and_get_aggregation()`, or `clear()`.

For edge nodes, the same aggregator is called **twice per round** with different model lists:

1. **First call:** `ctx.aggregator.aggregate(worker_models + [own_model])` — aggregates worker models with the edge's own trained model.
2. **Second call:** `ctx.aggregator.aggregate(edge_models)` — aggregates edge-to-edge models during gossip.

Model collection and synchronization are handled entirely by the stage logic:
- Models are stored in `ctx.peers[source].model` as they arrive via `@on_message` handlers
- Completion is tracked with `asyncio.Event` instances (`_workers_complete`, `_edges_complete`)
- Between phases, `ctx.peers` is reset and the model's contributors are flattened from `[edgeA, w1, w2, ...]` to `[edgeA]`

**Critical:** The contributor flattening between phases is essential. Without it, the inter-edge gossip cannot correctly track which edge models have been received, since worker addresses would appear in the contributors list (see section 4.4.5).

### 5.4 Dedicated @on_message handlers for inter-edge gossip

For the global aggregation phase, edge nodes use a dedicated `@on_message("edge_model", during={"edge_gossip"})` handler instead of reusing existing partial model handlers. This provides:

1. **Handler isolation:** The `during={"edge_gossip"}` parameter ensures the handler only activates during the gossip stage, avoiding conflicts
2. **Type-safe context:** Handlers access `self.ctx` (typed as `HFLContext`) for clean state management
3. **Tracking via peers dict:** Uses `ctx.peers[source].aggregated_from` instead of shared state dictionaries

The `gossip_weights` transport layer itself IS reused — it accepts generic callbacks (`early_stopping_fn`, `get_candidates_fn`, `status_fn`, `model_fn`) that the `HFLEdgeGossipStage` implements with HFL-specific logic scoped to the typed context.

### 5.5 Worker-edge communication is direct (not gossip)

Workers send models directly to their edge node via `await ctx.cp.send()`. Similarly, edges push global models back to workers via direct send. This is point-to-point and more efficient than gossip for the intra-cluster communication where the target is known.

---

## 6. File Change Summary

### Existing files modified

| File | Change |
|---|---|
| `p2pfl/utils/topologies.py` | Add `HIERARCHICAL` enum value and async `connect_hierarchical()` method |
| `p2pfl/workflow/factory.py` | Register HFL workflow: `register_workflow("hfl", HFL)` |

### New files

| File | Description |
|---|---|
| `p2pfl/workflow/hfl/__init__.py` | Package init |
| `p2pfl/workflow/hfl/context.py` | `HFLContext` and `HFLPeerState` dataclasses |
| `p2pfl/workflow/hfl/workflow.py` | `HFL` workflow class |
| `p2pfl/workflow/hfl/stages/__init__.py` | Stages package |
| `p2pfl/workflow/hfl/stages/setup.py` | Setup stage (branching) |
| `p2pfl/workflow/hfl/stages/worker_train.py` | Worker training |
| `p2pfl/workflow/hfl/stages/worker_wait_global.py` | Worker wait for global model |
| `p2pfl/workflow/hfl/stages/edge_local_train.py` | Edge local training |
| `p2pfl/workflow/hfl/stages/edge_aggregate_workers.py` | Edge aggregates workers + flatten |
| `p2pfl/workflow/hfl/stages/edge_gossip.py` | Inter-edge gossip |
| `p2pfl/workflow/hfl/stages/edge_distribute.py` | Edge distributes to workers |
| `p2pfl/workflow/hfl/stages/round_finished.py` | Round completion (shared) |

**NOTE:** There are NO command files (`hfl_worker_model_command.py`, etc.). Message handling is done via `@on_message` decorators in stages.

---

## 7. Implementation Order

1. `HFLContext` and `HFLPeerState` dataclasses (`p2pfl/workflow/hfl/context.py`)
2. Stages individual files (`p2pfl/workflow/hfl/stages/*.py`) - 8 total
3. `HFL` workflow class (`p2pfl/workflow/hfl/workflow.py`)
4. Topology helpers (`p2pfl/utils/topologies.py` - add `connect_hierarchical()`)
5. Factory registration in `p2pfl/workflow/factory.py`: `register_workflow("hfl", HFL)`
6. Tests

---

## 8. Potential Challenges

### 8.1 Timing and synchronization

Workers must wait for the edge to finish local aggregation AND inter-edge gossip before receiving the global model. The `ctx.global_model_ready` asyncio event handles this with `await asyncio.wait_for()`, but timeout values need careful tuning. If an edge is slow, workers will timeout.

### 8.2 Contributor flattening between aggregation phases

After local aggregation, the edge model has `contributors=[edgeA, w1, w2, ...]`. This MUST be flattened to `[edgeA]` before the inter-edge gossip phase. If this step is missed, the edge gossip stage cannot correctly track which edge models have been received — `ctx.peers` is keyed by edge address, but `_all_edge_models_received()` expects only edge addresses in the contributor lists. This is the most subtle bug-prone aspect of the design.

### 8.3 Learning start propagation

When `set_start_learning` is called on an edge, the `StartLearningCommand` broadcasts to all neighbors (including workers). The `HFLStartLearningStage` must ensure workers also enter the learning workflow correctly. This requires the `StartLearningCommand` to propagate through the hierarchy.

### 8.4 Round tracking between tiers

`nei_status` tracking may need adjustment since workers and edges are at different points in their workflows during a round. The `HFLEdgeGossipStage` only uses `hfl_context.edge_models_aggregated` for inter-edge gossip candidates (not `nei_status`), so this should remain isolated from worker status.

### 8.5 Peer state reset between aggregation phases

The edge node uses `ctx.peers` for two sequential aggregation phases (worker aggregation, then edge gossip). The `ctx.peers` dict must be properly reset between phases — if stale worker models remain in `ctx.peers` during the edge gossip phase, they would be included in the `aggregator.aggregate()` call and corrupt the global model. The `HFLEdgeGossipStage.run()` method handles this by calling `peer.reset_round()` on all peers before storing the edge-level model.

### 8.6 Async/await correctness

All I/O operations in the new architecture must use `await`. This includes:
- `await ctx.learner.fit()`
- `await ctx.cp.send()`
- `await ctx.cp.connect()`
- `await asyncio.wait_for(ctx.global_model_ready.wait(), timeout=...)`

Forgetting `await` on async operations will cause runtime errors or silent failures. The workflow engine validates that all stage `run()` methods are properly async.

---

## 9. Test Plan

### 9.1 Integration test

**File:** `test/hfl_test.py`

```python
import pytest
from p2pfl.workflow.factory import create_workflow

@pytest.mark.asyncio
async def test_hfl_convergence():
    """Test hierarchical FL with 2 edges, 2 workers each."""
    data = P2PFLDataset.from_huggingface("p2pfl/MNIST")
    partitions = data.generate_partitions(6 * 50, RandomIIDPartitionStrategy)

    # Create edge nodes
    e1 = Node(model_fn(), partitions[0])
    e1.workflow = create_workflow("hfl")
    e2 = Node(model_fn(), partitions[1])
    e2.workflow = create_workflow("hfl")

    # Create worker nodes
    w1 = Node(model_fn(), partitions[2])
    w1.workflow = create_workflow("hfl")
    w2 = Node(model_fn(), partitions[3])
    w2.workflow = create_workflow("hfl")
    w3 = Node(model_fn(), partitions[4])
    w3.workflow = create_workflow("hfl")
    w4 = Node(model_fn(), partitions[5])
    w4.workflow = create_workflow("hfl")

    all_nodes = [e1, e2, w1, w2, w3, w4]

    # Start nodes
    for n in all_nodes:
        await n.start()

    # Setup hierarchy: w1,w2 -> e1; w3,w4 -> e2; e1 <-> e2
    await TopologyFactory.connect_hierarchical(
        edge_nodes=[e1, e2],
        worker_groups=[[w1, w2], [w3, w4]],
    )

    await e1.set_start_learning(rounds=2, epochs=1)
    await wait_to_finish(all_nodes, timeout=240)

    # Verify model convergence
    check_equal_models(all_nodes)

    # Stop nodes
    for n in all_nodes:
        await n.stop()
```

### 9.2 Verification strategy

1. **HFL integration test:** 2 edges, 2 workers per edge, MNIST, verify all nodes finish and models converge.
2. **Regression:** Run the full existing test suite (`pytest`) to verify the P2P workflow remains unchanged (no modifications to `Node`, `NodeState`, or existing stages).
3. **Flow validation:** Verify via `learning_workflow.history` that the correct stages execute for each role in the correct order.

---

## 10. Example Usage

```python
import asyncio
from p2pfl.node import Node
from p2pfl.workflow.factory import create_workflow
from p2pfl.utils.topologies import TopologyFactory


async def main():
    # Create nodes
    edge1 = Node(model, data_edge1)
    edge2 = Node(model, data_edge2)
    worker1 = Node(model, data_w1)
    worker2 = Node(model, data_w2)
    worker3 = Node(model, data_w3)
    worker4 = Node(model, data_w4)

    # Assign HFL workflows via factory
    edge1.workflow = create_workflow("hfl")
    edge2.workflow = create_workflow("hfl")
    worker1.workflow = create_workflow("hfl")
    worker2.workflow = create_workflow("hfl")
    worker3.workflow = create_workflow("hfl")
    worker4.workflow = create_workflow("hfl")

    # Start all nodes
    for n in [edge1, edge2, worker1, worker2, worker3, worker4]:
        await n.start()

    # Connect hierarchy: workers -> edges, edges <-> edges
    # This also configures each workflow's context
    await TopologyFactory.connect_hierarchical(
        edge_nodes=[edge1, edge2],
        worker_groups=[[worker1, worker2], [worker3, worker4]],
    )

    # Start learning from any edge
    await edge1.set_start_learning(rounds=5, epochs=2)


# Run
asyncio.run(main())
```
