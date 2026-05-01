# Test Coverage Map

Module-by-module mapping between source code and tests.
Run `uv run pytest` for all tests, or target a specific module with `uv run pytest test/<path>`.

Regenerate coverage dashboard: `uv run coverage json && uv run python scripts/coverage_dashboard.py`

## communication/

| Source | Cover | Test File |
|--------|-------|-----------|
| `commands/command.py` | 100% | `communication/commands_test.py` |
| `commands/infrastructure/*.py` | 100% | `communication/commands_test.py` |
| `commands/workflow/workflow_command.py` | 91% | `communication/commands_test.py` |
| `protocols/communication_protocol.py` | 74% | `communication/communication_test.py` |
| `protocols/protobuff/gossiper.py` | 68% | `communication/gossip_propagation_test.py` |
| `protocols/protobuff/heartbeater.py` | 91% | `communication/communication_test.py` |
| `protocols/protobuff/neighbors.py` | 91% | `communication/communication_test.py` |
| `protocols/protobuff/server.py` | 80% | `communication/communication_test.py`, `communication/message_buffering_test.py` |
| `protocols/protobuff/client.py` | 85% | `communication/commands_test.py` |
| `protocols/protobuff/grpc/address.py` | 83% | `communication/commands_test.py` |
| `protocols/protobuff/grpc/client.py` | 78% | `communication/commands_test.py`, `communication/communication_test.py` |
| `protocols/protobuff/grpc/server.py` | 78% | `communication/communication_test.py` |
| `protocols/protobuff/memory/*.py` | 77-100% | `communication/communication_test.py`, `communication/gossip_propagation_test.py` |
| `protocols/protobuff/protobuff_communication_protocol.py` | 89% | `communication/communication_test.py` |
| `protocols/protobuff/proto/generate_proto.py` | 0% | **none** (codegen script) |

**Additional tests:** `communication/pre_send_model_command_test.py`

## learning/aggregators/

| Source | Cover | Test File |
|--------|-------|-----------|
| `aggregator.py` | 94% | `learning/aggregators/base_test.py` |
| `fedavg.py` | 100% | `learning/aggregators/weight_aggregators_test.py` |
| `fedmedian.py` | 94% | `learning/aggregators/weight_aggregators_test.py` |
| `fedprox.py` | 100% | `learning/aggregators/weight_aggregators_test.py` |
| `fedopt/*.py` | 97-100% | `learning/aggregators/weight_aggregators_test.py` |
| `krum.py` | 95% | `learning/aggregators/weight_aggregators_test.py` |
| `scaffold.py` | 88% | `learning/aggregators/weight_aggregators_test.py` |
| `sequential.py` | 93% | `learning/aggregators/base_test.py` |
| `fedxgbbagging.py` | 93% | `learning/aggregators/tree_aggregators_test.py` |
| `pushsum.py` | **23%** | **needs tests** |

## learning/compression/

| Source | Cover | Test File |
|--------|-------|-----------|
| `dp_strategy.py` | 84% | `learning/dp_test.py` |
| `quantization_strategy.py` | 68% | `learning/compression_test.py` |
| `topk_strategy.py` | 96% | `learning/compression_test.py` |
| `lra_strategy.py` | 100% | `learning/compression_test.py` |
| `zlib_strategy.py` | 100% | `learning/compression_test.py` |
| `lzma_strategy.py` | 100% | `learning/compression_test.py` |
| `manager.py` | 93% | `learning/compression_test.py` |

## learning/frameworks/

| Source | Cover | Test File |
|--------|-------|-----------|
| `p2pfl_model.py` | 86% | `learning/frameworks/weight_frameworks_test.py`, `learning/frameworks/tree_frameworks_test.py` |
| `learner.py` | 76% | `learning/frameworks/weight_frameworks_test.py` |
| `callback.py` | 92% | `learning/callbacks_test.py` |
| `callback_factory.py` | 85% | `learning/callbacks_test.py` |
| **pytorch/** | | |
| `lightning_model.py` | 93% | `learning/frameworks/weight_frameworks_test.py` |
| `lightning_learner.py` | 81% | `learning/frameworks/weight_frameworks_test.py` |
| `callbacks/fedprox_callback.py` | 89% | `learning/callbacks_test.py` |
| `callbacks/scaffold_callback.py` | **40%** | **needs tests** |
| **tensorflow/** | | |
| `keras_model.py` | 94% | `learning/frameworks/weight_frameworks_test.py` |
| `keras_learner.py` | 65% | `learning/frameworks/weight_frameworks_test.py` |
| `callbacks/scaffold_callback.py` | **26%** | **needs tests** |
| `custom_models/asydfl_model.py` | **0%** | **none** |
| **xgboost/** | | |
| `xgboost_model.py` | 81% | `learning/frameworks/tree_frameworks_test.py` |
| `xgboost_learner.py` | 83% | `learning/frameworks/tree_frameworks_test.py` |
| `xgboost_logger.py` | **0%** | **none** |
| **flax/** | | |
| `flax_model.py` | **37%** | **needs tests** |
| `flax_learner.py` | **0%** | **none** |

## learning/dataset/

| Source | Cover | Test File |
|--------|-------|-----------|
| `p2pfl_dataset.py` | 72% | `learning/p2pfl_dataset_test.py` |
| `partition_strategies.py` | 88% | `learning/p2pfl_dataset_test.py` |

## management/

| Source | Cover | Test File |
|--------|-------|-----------|
| `cli.py` | ~85% | `management/management_test.py` |
| `message_storage.py` | ~95% | `management/management_test.py` |
| `metric_storage.py` | ~95% | `management/management_test.py` |
| `node_monitor.py` | ~70% | `management/management_test.py` |
| `logger/logger.py` | 82% | (exercised by e2e tests) |
| `logger/decorators/wandb_logger.py` | **46%** | **needs tests** (requires mocking wandb) |
| `logger/decorators/web_logger.py` | **36%** | **needs tests** (requires mocking httpx) |
| `p2pfl_web_services.py` | **20%** | `management/web_test.py` |
| `launch_from_yaml/*.py` | **0%** | **none** |

## workflow/

| Source | Cover | Test File |
|--------|-------|-----------|
| `engine/workflow.py` | 78% | `workflow/base_test.py`, `node_test.py` (e2e) |
| `engine/experiment.py` | 92% | `workflow/base_test.py` |
| `engine/observable.py` | 100% | `workflow/base_test.py` |
| `engine/stage.py` | 100% | `workflow/base_test.py` |
| `engine/message.py` | 93% | `workflow/basic_dfl_test.py` |
| `factory.py` | 100% | `workflow/factory_test.py` |
| `validation.py` | 64% | `workflow/basic_dfl_test.py`, `workflow/async_dfl_test.py` |
| **basic_dfl/** | | |
| `workflow.py` | 93% | `workflow/basic_dfl_test.py` |
| `stages/setup.py` | 93% | `node_test.py` (e2e) |
| `stages/learning_gossip_loop.py` | 83% | `node_test.py` (e2e) |
| `stages/learning_wait_model.py` | **38%** | **needs tests** |
| `stages/round_init.py` | 79% | `node_test.py` (e2e) |
| `stages/voting.py` | 83% | `node_test.py` (e2e) |
| **async_dfl/** | | |
| `stages/setup.py` | **30%** | **needs tests** |
| `stages/training_round.py` | **20%** | **needs tests** |
| `diagram.py` | **0%** | **none** (visualization utility) |

## node & utils

| Source | Cover | Test File |
|--------|-------|-----------|
| `node.py` | 75% | `node_test.py` |
| `node_state.py` | 77% | `node_test.py` |
| `settings.py` | 92% | (used everywhere) |
| `utils/topologies.py` | 97% | `utils_test.py` |
| `utils/node_component.py` | 96% | `utils_test.py` |
| `utils/utils.py` | 79% | `node_test.py`, `reproducibility_test.py` |
| `utils/seed.py` | 64% | `reproducibility_test.py` |
| `__main__.py` | ~100% | `management/management_test.py` (tests `app` import) |

## Priority gaps (by impact)

1. **pushsum.py** (23%, 17 stmts) — small, easy win
2. **learning_wait_model.py** (38%, 20 stmts) — small
3. **scaffold callbacks** (PT 40%, TF 26%) — need mocked learner tests
4. **async_dfl stages** (20-30%) — need async workflow tests
5. **web_logger / wandb_logger** (36-46%) — need mocked external service tests
6. **launch_from_yaml** (0%, 259 stmts) — large, consider partial coverage
7. **flax** (0-37%) — depends on jax availability
