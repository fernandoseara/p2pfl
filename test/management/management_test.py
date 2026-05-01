#
# This file is part of the p2pfl distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2024 Pedro Guijas Bravo.
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
"""Tests for p2pfl management: CLI, MessageStorage, NodeMonitor, MetricStorage."""

import asyncio
import os
import subprocess
import types
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from p2pfl.management.cli import app
from p2pfl.management.message_storage import MessageStorage
from p2pfl.management.metric_storage import GlobalMetricStorage, LocalMetricStorage
from p2pfl.management.node_monitor import (
    NodeMonitor,
    _detect_gpus,
    _get_geolocation,
    collect_node_metadata,
)

runner = CliRunner()


###
# CLI
###


class TestCLI:
    """CLI tests."""

    def test_help(self):
        """Help flag shows usage info."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "P2PFL" in result.output

    def test_remote_not_implemented(self):
        """Remote command returns success."""
        result = runner.invoke(app, ["remote"])
        assert result.exit_code == 0

    def test_login_rejects_empty_credentials(self):
        """Login rejects empty credentials."""
        assert runner.invoke(app, ["login", "--url", "", "--token", "tok"]).exit_code == 1
        assert runner.invoke(app, ["login", "--url", "http://x", "--token", ""]).exit_code == 1

    def test_login_success_writes_env_file(self, tmp_path):
        """Login writes credentials to env file."""
        env_file = str(tmp_path / ".p2pfl_env")
        with patch("p2pfl.management.cli.os.path.join", return_value=env_file):
            result = runner.invoke(app, ["login", "--url", "http://test", "--token", "secret"])
        assert result.exit_code == 0
        assert "Successfully authenticated" in result.output
        assert os.path.exists(env_file)
        with open(env_file) as f:
            content = f.read()
        assert "http://test" in content
        assert "secret" in content

    def test_login_warns_on_unwritable_path(self, tmp_path):
        """Login warns on unwritable path."""
        bad_path = str(tmp_path / "nonexistent" / "dir" / ".p2pfl_env")
        with patch("p2pfl.management.cli.os.path.join", return_value=bad_path):
            result = runner.invoke(app, ["login", "--url", "http://test", "--token", "secret"])
        assert result.exit_code == 0
        assert "Warning" in result.output

    def test_run_rejects_missing_yaml(self):
        """Run rejects missing YAML file."""
        result = runner.invoke(app, ["run", "/nonexistent/path.yaml"])
        assert result.exit_code == 1

    def test_run_rejects_unknown_example(self):
        """Run rejects unknown example name."""
        result = runner.invoke(app, ["run", "nonexistent_example_name_xyz"])
        assert result.exit_code == 1

    def test_list_examples(self):
        """List examples returns success."""
        result = runner.invoke(app, ["list-examples"])
        assert result.exit_code == 0


###
# MessageStorage
###


class TestMessageStorage:
    """MessageStorage tests."""

    def test_sent_sets_source_and_destination(self):
        """Sent message sets source and destination."""
        s = MessageStorage(disable_locks=True)
        s.add_message("A", "sent", "hello", "B", "message", 100, additional_info={"k": "v"})
        msg = s.get_messages()[0]
        assert msg["source"] == "A"
        assert msg["destination"] == "B"
        assert msg["additional_info"] == {"k": "v"}

    def test_received_reverses_source_and_destination(self):
        """Received message reverses source and destination."""
        s = MessageStorage(disable_locks=True)
        s.add_message("A", "received", "hello", "B", "message", 200)
        msg = s.get_messages()[0]
        assert msg["source"] == "B"
        assert msg["destination"] == "A"

    def test_rejects_invalid_direction(self):
        """Invalid direction raises ValueError."""
        s = MessageStorage(disable_locks=True)
        with pytest.raises(ValueError, match="Invalid direction"):
            s.add_message("A", "invalid", "cmd", "B", "message", 0)
        with pytest.raises(ValueError, match="Invalid direction"):
            s.get_messages(direction="invalid")

    def test_filter_by_direction_and_node(self):
        """Messages can be filtered by direction and node."""
        s = MessageStorage(disable_locks=True)
        s.add_message("A", "sent", "cmd1", "B", "message", 10)
        # C received from A => source=A, destination=C
        s.add_message("C", "received", "cmd2", "A", "message", 20)
        assert len(s.get_sent_messages()) == 1
        assert len(s.get_received_messages()) == 1
        # node=A appears as source in both messages
        assert len(s.get_messages(node="A")) == 2
        assert len(s.get_sent_messages(node="A")) == 1
        # C is the destination of the received message
        assert len(s.get_received_messages(node="C")) == 1

    def test_filter_by_cmd_and_round(self):
        """Messages can be filtered by command and round."""
        s = MessageStorage(disable_locks=True)
        s.add_message("A", "sent", "hello", "B", "message", 10, round_num=1)
        s.add_message("A", "sent", "bye", "B", "message", 10, round_num=2)
        assert len(s.get_messages(cmd="hello")) == 1
        assert len(s.get_messages(round_num=2)) == 1

    def test_limit_returns_most_recent(self):
        """Limit returns most recent messages."""
        s = MessageStorage(disable_locks=True)
        for i in range(10):
            s.add_message("A", "sent", f"cmd{i}", "B", "message", i)
        result = s.get_messages(limit=3)
        assert len(result) == 3
        assert result[0]["cmd"] == "cmd7"

    def test_thread_safe_with_locks(self):
        """Thread-safe mode works with locks enabled."""
        s = MessageStorage(disable_locks=False)
        s.add_message("A", "sent", "cmd", "B", "message", 10)
        assert len(s.get_messages()) == 1


###
# LocalMetricStorage
###


class TestLocalMetricStorage:
    """LocalMetricStorage tests."""

    def test_log_hierarchy(self):
        """Logs stored in experiment/round/node hierarchy."""
        s = LocalMetricStorage(disable_locks=True)
        s.add_log("exp1", 0, "loss", "n1", 0.5, step=1)
        s.add_log("exp1", 0, "loss", "n1", 0.3, step=2)
        s.add_log("exp1", 1, "acc", "n2", 0.9, step=0)

        assert "exp1" in s.get_all_logs()
        assert len(s.get_experiment_logs("exp1")) == 2  # rounds 0 and 1
        assert "n1" in s.get_experiment_round_logs("exp1", 0)
        assert s.get_experiment_round_node_logs("exp1", 0, "n1")["loss"] == [(1, 0.5), (2, 0.3)]

    def test_thread_safe_with_locks(self):
        """Thread-safe mode works with locks enabled."""
        s = LocalMetricStorage(disable_locks=False)
        s.add_log("exp1", 0, "loss", "n1", 0.5, step=0)
        assert s.get_all_logs()["exp1"][0]["n1"]["loss"] == [(0, 0.5)]


###
# GlobalMetricStorage
###


class TestGlobalMetricStorage:
    """GlobalMetricStorage tests."""

    def test_deduplicates_same_round(self):
        """Same round metric is deduplicated."""
        s = GlobalMetricStorage(disable_locks=True)
        s.add_log("exp1", 0, "acc", "n1", 0.8)
        s.add_log("exp1", 0, "acc", "n1", 0.9)
        assert len(s.get_experiment_node_logs("exp1", "n1")["acc"]) == 1

    def test_appends_different_rounds(self):
        """Different rounds are appended."""
        s = GlobalMetricStorage(disable_locks=True)
        s.add_log("exp1", 0, "acc", "n1", 0.8)
        s.add_log("exp1", 1, "acc", "n1", 0.9)
        logs = s.get_experiment_logs("exp1")
        assert "n1" in logs
        assert len(logs["n1"]["acc"]) == 2

    def test_thread_safe_with_locks(self):
        """Thread-safe mode works with locks enabled."""
        s = GlobalMetricStorage(disable_locks=False)
        s.add_log("exp1", 0, "loss", "n1", 0.5)
        assert s.get_all_logs()["exp1"]["n1"]["loss"] == [(0, 0.5)]


###
# NodeMonitor
###


class TestNodeMonitor:
    """NodeMonitor tests."""

    def test_report_system_resources(self):
        """Report system resources returns cpu and ram."""
        m = NodeMonitor()
        res = m._report_system_resources()
        assert "cpu" in res
        assert "ram" in res
        assert isinstance(res["cpu"], float)

    @pytest.mark.asyncio
    async def test_lifecycle_collects_metrics_and_fires_callbacks(self):
        """Test lifecycle collects metrics and fires callbacks."""
        collected = []
        m = NodeMonitor()
        m.period = 0.05
        m.add_callback(lambda dt, d: collected.append(d))
        m.start()
        assert m.running
        # start is idempotent
        task_ref = m._task
        m.start()
        assert m._task is task_ref

        await asyncio.sleep(0.15)
        m.stop()
        assert not m.running
        assert len(m.get_logs()) > 0
        assert len(collected) > 0
        assert "cpu" in collected[0]


###
# collect_node_metadata & helpers
###


class TestNodeMetadata:
    """Node metadata collection tests."""

    def setup_method(self):
        """Reset cached metadata before each test."""
        import p2pfl.management.node_monitor as nm

        nm._cached_metadata = None

    def test_collect_metadata_and_cache(self):
        """Metadata is collected and cached."""
        r1 = collect_node_metadata()
        assert "os" in r1 and "hardware" in r1 and "runtime" in r1
        r2 = collect_node_metadata()
        assert r1 is r2

    def test_detect_gpus(self):
        """Detect GPUs returns a list."""
        result = _detect_gpus()
        assert isinstance(result, list)

    def test_geolocation_disabled(self):
        """Geolocation returns None when disabled."""
        from p2pfl.settings import Settings

        Settings.general.WEB_GEOLOCATION = False
        assert _get_geolocation() is None

    # ── remove_callback ─────────────────────────────────────────────

    def test_remove_callback_registered(self):
        """Registered callback can be removed."""
        m = NodeMonitor()

        def cb(dt, d):
            pass

        m.add_callback(cb)
        assert cb in m._callbacks
        m.remove_callback(cb)
        assert cb not in m._callbacks

    def test_remove_callback_not_registered(self):
        """remove_callback on unknown callback should not raise."""
        m = NodeMonitor()
        m.remove_callback(lambda dt, d: None)  # no-op, must not raise

    # ── _report_system_resources: nvidia-smi success ────────────────

    def test_report_resources_nvidia_smi_success(self):
        """nvidia-smi success parses GPU utilization."""
        m = NodeMonitor()
        fake = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="45, 2048, 8192\n70, 4096, 8192\n",
        )
        with patch("p2pfl.management.node_monitor.subprocess.run", return_value=fake):
            res = m._report_system_resources()
        assert res["gpu0_util"] == 45.0
        assert res["gpu0_vram"] == pytest.approx((2048 / 8192) * 100)
        assert res["gpu1_util"] == 70.0

    def test_report_resources_nvidia_smi_zero_total(self):
        """Zero total memory should produce 0.0 vram, not a division error."""
        m = NodeMonitor()
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="50, 0, 0\n")
        with patch("p2pfl.management.node_monitor.subprocess.run", return_value=fake):
            res = m._report_system_resources()
        assert res["gpu0_vram"] == 0.0

    def test_report_resources_nvidia_smi_bad_csv(self):
        """Malformed lines (wrong number of columns) are silently skipped."""
        m = NodeMonitor()
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="only_one_field\n")
        with patch("p2pfl.management.node_monitor.subprocess.run", return_value=fake):
            res = m._report_system_resources()
        assert "gpu0_util" not in res

    # ── _detect_gpus: CUDA path ────────────────────────────────────

    def test_detect_gpus_cuda(self):
        """CUDA GPUs are detected via torch."""
        props = types.SimpleNamespace(name="RTX 4090", total_memory=24 * 1024 * 1024 * 1024)
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_properties.return_value = props
        with patch.dict("sys.modules", {"torch": mock_torch}):
            gpus = _detect_gpus()
        assert len(gpus) == 1
        assert gpus[0]["backend"] == "cuda"
        assert gpus[0]["name"] == "RTX 4090"
        assert gpus[0]["memory_mb"] == 24 * 1024

    def test_detect_gpus_cuda_import_error(self):
        """If torch import raises, fall through to nvidia-smi fallback."""

        def raise_import(*args, **kwargs):
            raise ImportError("no torch")

        with patch("builtins.__import__", side_effect=raise_import):
            # Cannot import torch at all, so both torch paths fail.
            # nvidia-smi also won't be available in CI, so expect empty.
            gpus = _detect_gpus()
        assert isinstance(gpus, list)

    # ── _detect_gpus: nvidia-smi fallback ──────────────────────────

    def test_detect_gpus_nvidia_smi_fallback(self):
        """When torch.cuda is unavailable, fall back to nvidia-smi."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        fake = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="Tesla V100, 16384\nTesla V100, 16384\n",
        )
        with patch.dict("sys.modules", {"torch": mock_torch}), patch("p2pfl.management.node_monitor.subprocess.run", return_value=fake):
            gpus = _detect_gpus()
        assert len(gpus) == 2
        assert gpus[0]["name"] == "Tesla V100"
        assert gpus[0]["memory_mb"] == 16384
        assert gpus[0]["backend"] == "cuda"

    def test_detect_gpus_nvidia_smi_bad_csv_fallback(self):
        """nvidia-smi returns malformed output: skip bad lines."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="only_name\n")
        with patch.dict("sys.modules", {"torch": mock_torch}), patch("p2pfl.management.node_monitor.subprocess.run", return_value=fake):
            gpus = _detect_gpus()
        # Bad line skipped, no gpus found, falls through to MPS check
        assert isinstance(gpus, list)

    # ── _detect_gpus: Apple MPS path ───────────────────────────────

    def test_detect_gpus_mps(self):
        """Apple MPS GPU is detected."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        # nvidia-smi must fail so we reach MPS
        fake = subprocess.CompletedProcess(args=[], returncode=1, stdout="")
        with patch.dict("sys.modules", {"torch": mock_torch}), patch("p2pfl.management.node_monitor.subprocess.run", return_value=fake):
            gpus = _detect_gpus()
        assert len(gpus) == 1
        assert gpus[0]["backend"] == "mps"
        assert gpus[0]["name"] == "Apple MPS"

    # ── _get_geolocation: enabled, success ─────────────────────────

    def test_geolocation_enabled_success(self):
        """Geolocation returns data when enabled and request succeeds."""
        from p2pfl.settings import Settings

        geo_data = {
            "country": "Spain",
            "regionName": "Galicia",
            "city": "Santiago",
            "lat": 42.88,
            "lon": -8.54,
            "isp": "Telco",
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = geo_data
        mock_httpx = MagicMock()
        mock_httpx.get.return_value = mock_resp

        Settings.general.WEB_GEOLOCATION = True
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = _get_geolocation()
        assert result is not None
        assert result["country"] == "Spain"
        assert result["lat"] == 42.88

    def test_geolocation_enabled_http_error(self):
        """Geolocation returns None on HTTP error."""
        from p2pfl.settings import Settings

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_httpx = MagicMock()
        mock_httpx.get.return_value = mock_resp

        Settings.general.WEB_GEOLOCATION = True
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = _get_geolocation()
        assert result is None

    def test_geolocation_enabled_exception(self):
        """Geolocation returns None on connection exception."""
        from p2pfl.settings import Settings

        mock_httpx = MagicMock()
        mock_httpx.get.side_effect = ConnectionError("no network")

        Settings.general.WEB_GEOLOCATION = True
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = _get_geolocation()
        assert result is None

    # ── collect_node_metadata: gpus and geolocation populated ──────

    def test_metadata_includes_gpus_and_geolocation(self):
        """Metadata includes GPU and geolocation data."""
        fake_gpus = [{"name": "GPU0", "memory_mb": 8192, "backend": "cuda"}]
        fake_geo = {"country": "US", "region": "CA", "city": "SF", "lat": 37.7, "lon": -122.4, "isp": "ISP"}

        with (
            patch("p2pfl.management.node_monitor._detect_gpus", return_value=fake_gpus),
            patch("p2pfl.management.node_monitor._get_geolocation", return_value=fake_geo),
        ):
            result = collect_node_metadata()
        assert result["gpus"] == fake_gpus
        assert result["geolocation"] == fake_geo

    def test_metadata_exception_returns_empty_dict(self):
        """Metadata returns empty dict on exception."""
        with patch("p2pfl.management.node_monitor.psutil") as mock_psutil:
            mock_psutil.virtual_memory.side_effect = RuntimeError("broken")
            result = collect_node_metadata()
        assert result == {}
