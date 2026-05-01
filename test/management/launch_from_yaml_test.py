"""Tests for p2pfl/management/launch_from_yaml (YAML parsing, config validation, utilities)."""

import asyncio
import json
import os
import tempfile
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from datasets import Dataset, DatasetDict

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.management.launch_from_yaml import _find_web_services, run_from_yaml
from p2pfl.management.launch_from_yaml.utils import (
    export_experiment_data,
    load_by_package_and_name,
    resize_partitions,
)
from p2pfl.settings import Settings

# ── Helpers ─────────────────────────────────────────────────────────────

_MOD = "p2pfl.management.launch_from_yaml"


def _make_partition(train_size: int, test_size: int, batch_size: int = 1) -> P2PFLDataset:
    """Create a minimal P2PFLDataset partition with numeric data."""
    train = Dataset.from_dict({"x": list(range(train_size)), "y": [0] * train_size})
    test = Dataset.from_dict({"x": list(range(test_size)), "y": [1] * test_size})
    return P2PFLDataset(
        DatasetDict({"train": train, "test": test}),
        train_split_name="train",
        test_split_name="test",
        batch_size=batch_size,
        dataset_name="test_ds",
    )


def _minimal_yaml_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a valid minimal config dict for run_from_yaml."""
    cfg: dict[str, Any] = {
        "network": {
            "package": "p2pfl.communication.protocols.protobuff.memory",
            "protocol": "MemoryCommunicationProtocol",
            "nodes": 2,
            "topology": "ring",
        },
        "experiment": {
            "name": "test_exp",
            "rounds": 1,
            "epochs": 1,
            "trainset_size": 2,
            "workflow": "basic",
            "dataset": {
                "source": "huggingface",
                "name": "p2pfl/MNIST",
                "batch_size": 32,
                "partitioning": {
                    "package": "p2pfl.learning.dataset.partition_strategies",
                    "strategy": "RandomIIDPartitionStrategy",
                    "params": {},
                },
            },
            "model": {
                "package": "some.module",
                "model_build_fn": "build",
                "params": {},
            },
            "aggregator": {
                "package": "p2pfl.learning.aggregators.fedavg",
                "aggregator": "FedAvg",
                "params": {},
            },
        },
        "settings": {
            "general": {"log_level": "WARNING"},
            "SSL": {"use_ssl": False},
        },
    }
    if overrides:
        _deep_merge(cfg, overrides)
    return cfg


def _deep_merge(base: dict, updates: dict) -> dict:
    """Recursively merge *updates* into *base* in place."""
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _write_yaml(cfg: dict[str, Any], tmpdir: str) -> str:
    """Dump cfg to a YAML file and return its path."""
    path = os.path.join(tmpdir, "config.yaml")
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return path


@contextmanager
def _full_run_mocks(
    *,
    load_side_effect=None,
    partitions=None,
    extra_patches=None,
):
    """Stub the heavy dependencies of run_from_yaml; yields a dict of active mocks."""
    if partitions is None:
        partitions = [MagicMock(), MagicMock()]

    with (
        patch(f"{_MOD}.P2PFLDataset") as mock_ds,
        patch(f"{_MOD}.load_by_package_and_name") as mock_load,
        patch(f"{_MOD}.Node") as mock_node_cls,
        patch(f"{_MOD}.TopologyFactory") as mock_topo,
        patch(f"{_MOD}.wait_convergence", new_callable=AsyncMock) as mock_wc,
        patch(f"{_MOD}.wait_to_finish", new_callable=AsyncMock) as mock_wf,
    ):
        mock_dataset = MagicMock()
        mock_ds.from_huggingface.return_value = mock_dataset
        mock_ds.from_csv.return_value = mock_dataset
        mock_ds.from_json.return_value = mock_dataset
        mock_ds.from_parquet.return_value = mock_dataset
        mock_ds.from_pandas.return_value = mock_dataset
        mock_dataset.generate_partitions.return_value = partitions

        if load_side_effect:
            mock_load.side_effect = load_side_effect
        else:
            mock_load.return_value = MagicMock()

        node_inst = AsyncMock()
        node_inst.workflow = None
        mock_node_cls.return_value = node_inst

        mock_topo.generate_matrix.return_value = MagicMock()
        mock_topo.connect_nodes = AsyncMock()

        mocks = {
            "ds_cls": mock_ds,
            "dataset": mock_dataset,
            "load": mock_load,
            "node_cls": mock_node_cls,
            "node": node_inst,
            "topo": mock_topo,
            "wait_convergence": mock_wc,
            "wait_to_finish": mock_wf,
        }

        if extra_patches:
            with extra_patches as ep:
                mocks["extra"] = ep
                yield mocks
        else:
            yield mocks


def _run_yaml(cfg: dict[str, Any]) -> Any:
    """Write cfg to a temp YAML and run run_from_yaml to completion."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_yaml(cfg, tmpdir)
        return asyncio.get_event_loop().run_until_complete(run_from_yaml(path))


# ── load_by_package_and_name ────────────────────────────────────────────


class TestLoadByPackageAndName:
    """Load By Package And Name tests."""

    def test_loads_builtin_class(self):
        """Test loads builtin class."""
        cls = load_by_package_and_name("json", "JSONDecoder")
        assert cls is json.JSONDecoder

    def test_loads_nested_module_attr(self):
        """Test loads nested module attr."""
        cls = load_by_package_and_name("os.path", "join")
        assert cls is os.path.join

    def test_missing_module_raises(self):
        """Test missing module raises."""
        with pytest.raises(ModuleNotFoundError):
            load_by_package_and_name("nonexistent.module.xyz", "Foo")

    def test_missing_attribute_raises(self):
        """Test missing attribute raises."""
        with pytest.raises(AttributeError):
            load_by_package_and_name("json", "NonExistentThing")


# ── resize_partitions ──────────────────────────────────────────────────


class TestResizePartitions:
    """Resize Partitions tests."""

    def test_downsample_train(self):
        """Test downsample train."""
        part = _make_partition(train_size=100, test_size=20)
        result = resize_partitions([part], samples_per_node=50)
        assert len(result) == 1
        assert len(result[0]._data["train"]) == 50

    def test_upsample_train(self):
        """Test upsample train."""
        part = _make_partition(train_size=10, test_size=5)
        result = resize_partitions([part], samples_per_node=30)
        assert len(result[0]._data["train"]) == 30

    def test_same_size_unchanged(self):
        """Test same size unchanged."""
        part = _make_partition(train_size=20, test_size=10)
        result = resize_partitions([part], samples_per_node=20)
        assert len(result[0]._data["train"]) == 20

    def test_test_ratio_preserved(self):
        """Test test ratio preserved."""
        part = _make_partition(train_size=100, test_size=50)
        result = resize_partitions([part], samples_per_node=50)
        # Original ratio is 50/100 = 0.5, so test should be ~25
        assert len(result[0]._data["test"]) == 25

    def test_explicit_test_samples(self):
        """Test explicit test samples."""
        part = _make_partition(train_size=100, test_size=50)
        result = resize_partitions([part], samples_per_node=40, test_samples_per_node=10)
        assert len(result[0]._data["train"]) == 40
        assert len(result[0]._data["test"]) == 10

    def test_preserves_metadata(self):
        """Test preserves metadata."""
        part = _make_partition(train_size=20, test_size=5, batch_size=16)
        result = resize_partitions([part], samples_per_node=10)
        assert result[0].batch_size == 16
        assert result[0].dataset_name == "test_ds"
        assert result[0]._train_split_name == "train"
        assert result[0]._test_split_name == "test"

    def test_multiple_partitions(self):
        """Test multiple partitions."""
        parts = [_make_partition(train_size=50, test_size=10) for _ in range(3)]
        result = resize_partitions(parts, samples_per_node=20)
        assert len(result) == 3
        for r in result:
            assert len(r._data["train"]) == 20

    def test_test_at_least_one(self):
        """When the test ratio is very small, test should still have at least 1 sample."""
        part = _make_partition(train_size=1000, test_size=1)
        result = resize_partitions([part], samples_per_node=10)
        assert len(result[0]._data["test"]) >= 1

    def test_deterministic_with_seed(self):
        """Resizing with the same seed produces identical results."""
        original_seed = Settings.general.SEED
        try:
            Settings.general.SEED = 42
            part1 = _make_partition(train_size=10, test_size=5)
            part2 = _make_partition(train_size=10, test_size=5)
            r1 = resize_partitions([part1], samples_per_node=30)
            r2 = resize_partitions([part2], samples_per_node=30)
            assert list(r1[0]._data["train"]["x"]) == list(r2[0]._data["train"]["x"])
        finally:
            Settings.general.SEED = original_seed


# ── export_experiment_data ─────────────────────────────────────────────


class TestExportExperimentData:
    """Export Experiment Data tests."""

    def test_creates_directory(self):
        """Test creates directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = os.path.join(tmpdir, "nested", "exp")
            export_experiment_data(exp_dir, [], [], None)
            assert os.path.isdir(exp_dir)

    def test_writes_stage_timings(self):
        """Test writes stage timings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = os.path.join(tmpdir, "exp")
            timings = [{"stage": "train", "duration": 1.5}]
            export_experiment_data(exp_dir, timings, [], None)
            path = os.path.join(exp_dir, "stage_timings.json")
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert data == timings

    def test_skips_empty_timings(self):
        """Test skips empty timings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = os.path.join(tmpdir, "exp")
            export_experiment_data(exp_dir, [], [], None)
            assert not os.path.exists(os.path.join(exp_dir, "stage_timings.json"))

    def test_writes_communication_logs(self):
        """Test writes communication logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = os.path.join(tmpdir, "exp")
            msgs = [{"type": "model", "size": 1024}]
            export_experiment_data(exp_dir, [], msgs, None)
            path = os.path.join(exp_dir, "communication_logs.json")
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert data == msgs

    def test_skips_empty_messages(self):
        """Test skips empty messages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = os.path.join(tmpdir, "exp")
            export_experiment_data(exp_dir, [], [], None)
            assert not os.path.exists(os.path.join(exp_dir, "communication_logs.json"))

    def test_writes_global_metrics_csv(self):
        """Test writes global metrics csv."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = os.path.join(tmpdir, "exp")
            metrics = {
                "exp1": {
                    "node_a": {
                        "loss": [(1, 0.5), (2, 0.3)],
                        "accuracy": [(1, 0.7)],
                    }
                }
            }
            export_experiment_data(exp_dir, [], [], metrics)
            csv_path = os.path.join(exp_dir, "global_metrics.csv")
            assert os.path.exists(csv_path)
            import pandas as pd

            df = pd.read_csv(csv_path)
            assert len(df) == 3
            assert set(df.columns) == {"experiment", "node", "metric", "round", "value"}
            assert df[df["metric"] == "loss"]["value"].tolist() == [0.5, 0.3]

    def test_skips_empty_global_metrics(self):
        """Test skips empty global metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = os.path.join(tmpdir, "exp")
            export_experiment_data(exp_dir, [], [], None)
            assert not os.path.exists(os.path.join(exp_dir, "global_metrics.csv"))

    def test_no_csv_for_empty_flat_metrics(self):
        """If global_metrics dict exists but has no actual metric entries, no CSV is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = os.path.join(tmpdir, "exp")
            export_experiment_data(exp_dir, [], [], {"exp1": {"node_a": {}}})
            assert not os.path.exists(os.path.join(exp_dir, "global_metrics.csv"))

    def test_multiple_experiments_and_nodes(self):
        """Test multiple experiments and nodes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = os.path.join(tmpdir, "exp")
            metrics = {
                "exp1": {
                    "nodeA": {"loss": [(1, 0.9)]},
                    "nodeB": {"loss": [(1, 0.8)]},
                },
                "exp2": {
                    "nodeC": {"acc": [(2, 0.95)]},
                },
            }
            export_experiment_data(exp_dir, [], [], metrics)
            import pandas as pd

            df = pd.read_csv(os.path.join(exp_dir, "global_metrics.csv"))
            assert len(df) == 3
            assert set(df["experiment"].unique()) == {"exp1", "exp2"}


# ── _find_web_services ─────────────────────────────────────────────────


class TestFindWebServices:
    """Find Web Services tests."""

    def test_returns_none_for_plain_logger(self):
        """Test returns none for plain logger."""
        lgr = MagicMock(spec=[])
        del lgr._p2pfl_web_services  # ensure attr doesn't exist
        lgr._p2pfl_logger = None
        assert _find_web_services(lgr) is None

    def test_finds_direct_web_services(self):
        """Test finds direct web services."""
        ws = MagicMock()
        lgr = MagicMock()
        lgr._p2pfl_web_services = ws
        assert _find_web_services(lgr) is ws

    def test_walks_decorator_chain(self):
        """Test walks decorator chain."""
        ws = MagicMock()
        inner = MagicMock()
        inner._p2pfl_web_services = ws

        outer = MagicMock()
        outer._p2pfl_web_services = None
        outer._p2pfl_logger = inner

        assert _find_web_services(outer) is ws

    def test_returns_none_on_empty_chain(self):
        """Test returns none on empty chain."""
        outer = MagicMock(spec=[])
        outer._p2pfl_logger = None
        del outer._p2pfl_web_services
        assert _find_web_services(outer) is None

    def test_three_level_chain(self):
        """Test three level chain."""
        ws = MagicMock()
        level3 = MagicMock()
        level3._p2pfl_web_services = ws

        level2 = MagicMock()
        level2._p2pfl_web_services = None
        level2._p2pfl_logger = level3

        level1 = MagicMock()
        level1._p2pfl_web_services = None
        level1._p2pfl_logger = level2

        assert _find_web_services(level1) is ws


# ── run_from_yaml: config validation errors ─────────────────────────────


class TestRunFromYamlValidation:
    """Test that run_from_yaml raises on invalid/missing config sections."""

    def test_missing_network(self):
        """Test missing network."""
        cfg = _minimal_yaml_config()
        del cfg["network"]
        with pytest.raises(ValueError, match="network"):
            _run_yaml(cfg)

    def test_missing_network_nodes(self):
        """Test missing network nodes."""
        cfg = _minimal_yaml_config()
        del cfg["network"]["nodes"]
        with pytest.raises(ValueError, match="'n'"):
            _run_yaml(cfg)

    def test_missing_dataset(self):
        """Test missing dataset."""
        cfg = _minimal_yaml_config()
        del cfg["experiment"]["dataset"]
        with pytest.raises(ValueError, match="dataset"):
            _run_yaml(cfg)

    def test_missing_dataset_source(self):
        """Test missing dataset source."""
        cfg = _minimal_yaml_config()
        del cfg["experiment"]["dataset"]["source"]
        with pytest.raises(ValueError, match="source"):
            _run_yaml(cfg)

    def test_missing_dataset_name(self):
        """Test missing dataset name."""
        cfg = _minimal_yaml_config()
        del cfg["experiment"]["dataset"]["name"]
        with pytest.raises(ValueError, match="name"):
            _run_yaml(cfg)

    def test_missing_partitioning(self):
        """Test missing partitioning."""
        cfg = _minimal_yaml_config()
        del cfg["experiment"]["dataset"]["partitioning"]
        with _full_run_mocks(), pytest.raises(ValueError, match="partitioning"):
            _run_yaml(cfg)

    def test_missing_partition_package(self):
        """Test missing partition package."""
        cfg = _minimal_yaml_config()
        del cfg["experiment"]["dataset"]["partitioning"]["package"]
        with _full_run_mocks(), pytest.raises(ValueError, match="partition_strategy"):
            _run_yaml(cfg)

    def test_missing_model(self):
        """Test missing model."""
        cfg = _minimal_yaml_config()
        del cfg["experiment"]["model"]
        with _full_run_mocks(), pytest.raises(ValueError, match="model"):
            _run_yaml(cfg)

    def test_missing_model_package(self):
        """Test missing model package."""
        cfg = _minimal_yaml_config()
        del cfg["experiment"]["model"]["package"]
        with _full_run_mocks(), pytest.raises(ValueError, match="model"):
            _run_yaml(cfg)

    def test_missing_aggregator(self):
        """Test missing aggregator."""
        cfg = _minimal_yaml_config()
        del cfg["experiment"]["aggregator"]
        with _full_run_mocks(), pytest.raises(ValueError, match="aggregator"):
            _run_yaml(cfg)

    def test_missing_workflow(self):
        """Test missing workflow."""
        cfg = _minimal_yaml_config()
        del cfg["experiment"]["workflow"]
        with _full_run_mocks(), pytest.raises(ValueError, match="workflow"):
            _run_yaml(cfg)

    def test_missing_protocol(self):
        """Test missing protocol."""
        cfg = _minimal_yaml_config()
        del cfg["network"]["protocol"]
        with _full_run_mocks(), pytest.raises(ValueError, match="protocol"):
            _run_yaml(cfg)

    def test_missing_topology(self, capsys):
        """Topology validation happens after node creation; the error is caught internally."""
        cfg = _minimal_yaml_config()
        del cfg["network"]["topology"]
        with _full_run_mocks():
            _run_yaml(cfg)
        captured = capsys.readouterr()
        assert "topology" in captured.out.lower()


# ── run_from_yaml: dataset loading dispatch ──────────────────────────────


class TestRunFromYamlDatasetSources:
    """Test that run_from_yaml dispatches to the correct P2PFLDataset factory for each source."""

    def _run_with_source(self, source: str, expected_factory: str) -> None:
        cfg = _minimal_yaml_config({"experiment": {"dataset": {"source": source}}})
        with _full_run_mocks() as m:
            _run_yaml(cfg)
            getattr(m["ds_cls"], expected_factory).assert_called_once()

    def test_huggingface_source(self):
        """Test huggingface source."""
        self._run_with_source("huggingface", "from_huggingface")

    def test_csv_source(self):
        """Test csv source."""
        self._run_with_source("csv", "from_csv")

    def test_json_source(self):
        """Test json source."""
        self._run_with_source("json", "from_json")

    def test_parquet_source(self):
        """Test parquet source."""
        self._run_with_source("parquet", "from_parquet")

    def test_pandas_source(self):
        """Test pandas source."""
        self._run_with_source("pandas", "from_pandas")

    def test_custom_source(self):
        """Test custom source."""
        cfg = _minimal_yaml_config(
            {
                "experiment": {
                    "dataset": {
                        "source": "custom",
                        "package": "some.module",
                        "class": "MyDataset",
                        "params": {"foo": 1},
                    }
                }
            }
        )
        mock_custom_cls = MagicMock()
        mock_custom_ds = MagicMock()
        mock_custom_cls.return_value = mock_custom_ds
        mock_custom_ds.generate_partitions.return_value = [MagicMock(), MagicMock()]

        def side_effect(pkg, name):
            if pkg == "some.module" and name == "MyDataset":
                return mock_custom_cls
            return MagicMock()

        with _full_run_mocks(load_side_effect=side_effect):
            _run_yaml(cfg)
        mock_custom_cls.assert_called_once_with(foo=1)

    def test_unknown_source_returns_none(self, capsys):
        """An unrecognized source produces no dataset, and function returns early."""
        cfg = _minimal_yaml_config({"experiment": {"dataset": {"source": "unknown_source"}}})
        result = _run_yaml(cfg)
        assert result is None
        captured = capsys.readouterr()
        assert "without creating a dataset" in captured.out

    def test_custom_source_missing_package(self):
        """Test custom source missing package."""
        cfg = _minimal_yaml_config(
            {
                "experiment": {
                    "dataset": {
                        "source": "custom",
                        "class": "MyDataset",
                    }
                }
            }
        )
        with pytest.raises(ValueError, match="Missing package or class"):
            _run_yaml(cfg)


# ── run_from_yaml: node setup and lifecycle ──────────────────────────────


class TestRunFromYamlNodeLifecycle:
    """Test that nodes are created, started, connected, and stopped correctly."""

    def test_nodes_created_started_stopped(self):
        """Test nodes created started stopped."""
        cfg = _minimal_yaml_config()
        with _full_run_mocks() as m:
            _run_yaml(cfg)
            assert m["node_cls"].call_count == 2
            assert m["node"].start.await_count == 2
            assert m["node"].stop.await_count == 2

    def test_rounds_less_than_one_stops_nodes(self):
        """Test rounds less than one stops nodes."""
        cfg = _minimal_yaml_config({"experiment": {"rounds": 0}})
        with _full_run_mocks() as m:
            _run_yaml(cfg)
            # Nodes should still be stopped in the finally block
            assert m["node"].stop.await_count == 2


# ── run_from_yaml: settings application ──────────────────────────────────


class TestRunFromYamlSettings:
    """Run From Yaml Settings tests."""

    def test_settings_applied_from_yaml(self):
        """Test settings applied from yaml."""
        cfg = _minimal_yaml_config(
            {
                "settings": {
                    "general": {"grpc_timeout": 99.0},
                }
            }
        )
        original = Settings.general.GRPC_TIMEOUT
        try:
            # Settings are applied before dataset loading; the run will fail on HF download
            with _full_run_mocks():
                _run_yaml(cfg)
            assert Settings.general.GRPC_TIMEOUT == 99.0
        finally:
            Settings.general.GRPC_TIMEOUT = original


# ── run_from_yaml: batch_size, transforms, samples_per_node ────────────


class TestRunFromYamlDatasetOptions:
    """Run From Yaml Dataset Options tests."""

    def test_batch_size_set(self):
        """Test batch size set."""
        cfg = _minimal_yaml_config({"experiment": {"dataset": {"batch_size": 64}}})
        with _full_run_mocks() as m:
            _run_yaml(cfg)
            m["dataset"].set_batch_size.assert_called_once_with(64)

    def test_default_batch_size(self):
        """Test default batch size."""
        cfg = _minimal_yaml_config()
        del cfg["experiment"]["dataset"]["batch_size"]
        with _full_run_mocks() as m:
            _run_yaml(cfg)
            m["dataset"].set_batch_size.assert_called_once_with(1)

    def test_transforms_applied(self):
        """Test transforms applied."""
        cfg = _minimal_yaml_config(
            {
                "experiment": {
                    "dataset": {
                        "transforms": {
                            "package": "some.transforms",
                            "function": "MyTransform",
                            "params": {"resize": 28},
                        }
                    }
                }
            }
        )
        mock_transform_cls = MagicMock()
        partition1, partition2 = MagicMock(), MagicMock()

        def side_effect(pkg, name):
            if pkg == "some.transforms" and name == "MyTransform":
                return mock_transform_cls
            return MagicMock()

        with _full_run_mocks(load_side_effect=side_effect, partitions=[partition1, partition2]):
            _run_yaml(cfg)
        partition1.set_transforms.assert_called_once()
        partition2.set_transforms.assert_called_once()

    def test_resize_partitions_called(self):
        """Test resize partitions called."""
        cfg = _minimal_yaml_config(
            {
                "experiment": {
                    "dataset": {
                        "partitioning": {
                            "package": "p2pfl.learning.dataset.partition_strategies",
                            "strategy": "RandomIIDPartitionStrategy",
                            "params": {},
                            "samples_per_node": 50,
                            "test_samples_per_node": 10,
                        }
                    }
                }
            }
        )
        with (
            _full_run_mocks(),
            patch(f"{_MOD}.resize_partitions", return_value=[MagicMock(), MagicMock()]) as mock_resize,
        ):
            _run_yaml(cfg)
            mock_resize.assert_called_once()
            args = mock_resize.call_args[0]
            assert args[1] == 50  # samples_per_node
            assert args[2] == 10  # test_samples_per_node

    def test_reduced_dataset_multiplies_partitions(self):
        """When reduced_dataset is True, partition count = n * reduction_factor."""
        cfg = _minimal_yaml_config(
            {
                "experiment": {
                    "dataset": {
                        "partitioning": {
                            "package": "p2pfl.learning.dataset.partition_strategies",
                            "strategy": "RandomIIDPartitionStrategy",
                            "params": {},
                            "reduced_dataset": True,
                            "reduction_factor": 5,
                        }
                    }
                }
            }
        )
        with _full_run_mocks(partitions=[MagicMock() for _ in range(10)]) as m:
            _run_yaml(cfg)
            call_args = m["dataset"].generate_partitions.call_args
            assert call_args[0][0] == 10  # 2 nodes * 5 reduction_factor

    def test_transforms_missing_package_raises(self):
        """Test transforms missing package raises."""
        cfg = _minimal_yaml_config(
            {
                "experiment": {
                    "dataset": {
                        "transforms": {
                            "function": "MyTransform",
                        }
                    }
                }
            }
        )
        with _full_run_mocks(), pytest.raises(ValueError, match="transforms"):
            _run_yaml(cfg)


# ── run_from_yaml: export_results flow ───────────────────────────────────


class TestRunFromYamlExportResults:
    """Run From Yaml Export Results tests."""

    def test_export_on_success(self):
        """Test export on success."""
        cfg = _minimal_yaml_config({"export_results": True, "output_dir": "/tmp/p2pfl_test_out"})
        with (
            _full_run_mocks(),
            patch(f"{_MOD}.export_experiment_data") as mock_export,
            patch(f"{_MOD}.logger") as mock_logger,
        ):
            mock_logger.get_messages.return_value = []
            mock_logger.get_global_logs.return_value = {}
            _run_yaml(cfg)
            mock_export.assert_called_once()

    def test_no_export_on_failure(self, capsys):
        """When the experiment fails, results are NOT exported."""
        cfg = _minimal_yaml_config({"export_results": True})
        with (
            _full_run_mocks() as m,
            patch(f"{_MOD}.export_experiment_data") as mock_export,
            patch(f"{_MOD}.logger") as mock_logger,
        ):
            m["wait_convergence"].side_effect = RuntimeError("convergence failed")
            mock_logger.get_messages.return_value = []
            _run_yaml(cfg)
            mock_export.assert_not_called()
        captured = capsys.readouterr()
        assert "FAILED" in captured.out


# ── run_from_yaml: model_fn compression parameter ──────────────────────


class TestRunFromYamlModelCompression:
    """Run From Yaml Model Compression tests."""

    def test_compression_passed_to_model(self):
        """Test compression passed to model."""
        compression_cfg = {"ptq": {"dtype": "int8"}}
        cfg = _minimal_yaml_config(
            {
                "experiment": {
                    "model": {
                        "package": "some.module",
                        "model_build_fn": "build",
                        "params": {"hidden": 64},
                        "compression": compression_cfg,
                    }
                }
            }
        )
        mock_model_cls = MagicMock()

        def side_effect(pkg, name):
            if pkg == "some.module" and name == "build":
                return mock_model_cls
            return MagicMock()

        with _full_run_mocks(load_side_effect=side_effect):
            _run_yaml(cfg)
        assert mock_model_cls.call_count == 2
        call_kwargs = mock_model_cls.call_args_list[0][1]
        assert call_kwargs["compression"] == compression_cfg
        assert call_kwargs["hidden"] == 64
