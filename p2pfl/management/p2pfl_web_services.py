#
# This file is part of the federated_learning_p2p (p2pfl) distribution
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

"""Communication with P2PFL Web Services (via REST API)."""

from __future__ import annotations

import asyncio
import datetime
import json
import threading
from typing import Any

import httpx
import websockets

from p2pfl.settings import Settings


class P2pflWebServicesError(Exception):
    """
    P2PFL Web Services Error.

    Args:
        code: Error code.
        message: Error message.

    """

    def __init__(self, code: int, message: str) -> None:
        """Initialize the error."""
        self.code = code
        self.message = message
        super().__init__(f"Error {code}: {message}")


class P2pflWebServices:
    """
    Class that manages the communication with the p2pfl web services.

    All buffered data (logs, metrics, messages, experiment updates, system
    metrics) is accumulated in a single buffer and flushed as a single
    ``POST /batch`` request — either periodically by a background asyncio
    task or when the buffer reaches ``Settings.general.WEB_BATCH_SIZE``.

    Each entry in the batch carries a ``type`` field so the backend can
    dispatch accordingly.

    Immediate (non-batched) operations like ``register_node`` and
    ``create_experiment`` use synchronous HTTP helpers.

    Args:
        url: The base URL of the web services API.
        key: The API key to access the services.

    """

    def __init__(self, url: str, key: str) -> None:
        """Initialize the p2pfl web services."""
        self._base_url = url.rstrip("/")
        self._headers = {
            "Content-Type": "application/json",
            "x-api-key": key,
        }
        # Maps node address -> server-assigned node id
        self.node_id: dict[str, int] = {}
        # Maps experiment name -> server-assigned experiment id
        self._exp_id: dict[str, int] = {}
        # Maps node address -> experiment name
        self._node_exp: dict[str, str] = {}

        # Single batch buffer — lock needed: singleton logger is called from multiple threads
        self._buffer: list[dict] = []
        self._lock = threading.Lock()
        self._flush_thread: threading.Thread | None = None
        self._running = False

        # Sync HTTP client for immediate operations (register, unregister, sync flush)
        self._client = httpx.Client(headers=self._headers, timeout=5.0)

    # --- Lifecycle ---

    def _ensure_flush_thread(self) -> None:
        """Lazily start a dedicated background thread for the WebSocket flush loop."""
        if self._flush_thread is not None and self._flush_thread.is_alive():
            return
        self._running = True
        t = threading.Thread(target=self._run_flush_loop, daemon=True, name="p2pfl-ws-flush")
        t.start()
        self._flush_thread = t

    def _run_flush_loop(self) -> None:
        """Entry point for the flush thread — creates its own event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._flush_loop())
        finally:
            loop.close()

    async def _flush_loop(self) -> None:
        """Flush buffered entries over a persistent WebSocket with auto-reconnect."""
        ws_url = self._base_url.replace("http://", "ws://").replace("https://", "wss://") + "/batch/ws"
        api_key = self._headers["x-api-key"]

        while self._running:
            pending: list[dict] = []
            try:
                async with websockets.connect(ws_url) as ws:
                    # Authenticate
                    await ws.send(json.dumps({"api_key": api_key}))
                    auth = json.loads(await ws.recv())
                    if not auth.get("authenticated"):
                        raise ConnectionError(f"Auth failed: {auth}")

                    while self._running:
                        await asyncio.sleep(Settings.general.WEB_BATCH_INTERVAL)
                        pending = self._drain()
                        if not pending:
                            continue
                        while pending:
                            chunk = pending[: Settings.general.WEB_BATCH_SIZE]
                            await ws.send(json.dumps(chunk))
                            await ws.recv()  # ack
                            pending = pending[Settings.general.WEB_BATCH_SIZE :]
            except Exception as e:
                if pending:
                    with self._lock:
                        self._buffer[:0] = pending
                print(f"[P2PFL Web Services] WebSocket error: {e}, reconnecting in 2s...")
                await asyncio.sleep(2)

    def stop(self) -> None:
        """Stop the flush thread, flush remaining data, close client."""
        self._running = False
        if self._flush_thread is not None:
            self._flush_thread.join(timeout=5)
        self._flush_thread = None
        self.flush()
        self._client.close()

    # --- Batching internals ---

    def _enqueue(self, entry: dict) -> None:
        """Append an entry to the batch buffer. Never blocks."""
        with self._lock:
            self._buffer.append(entry)
            if len(self._buffer) > Settings.general.WEB_MAX_BUFFER_SIZE:
                overflow = len(self._buffer) - Settings.general.WEB_MAX_BUFFER_SIZE
                del self._buffer[:overflow]
        self._ensure_flush_thread()

    def _drain(self) -> list[dict]:
        """Atomically drain the buffer."""
        with self._lock:
            entries = self._buffer
            self._buffer = []
        return entries

    def flush(self) -> None:
        """Flush all buffered entries synchronously (used during shutdown)."""
        entries = self._drain()
        if entries:
            self._sync_send(entries)

    def _sync_send(self, entries: list[dict]) -> None:
        """Send a batch synchronously via ``POST /batch``. Best-effort (drops on failure)."""
        try:
            response = self._client.post(self._base_url + "/batch", json=entries, timeout=10)
            response.raise_for_status()
        except Exception as e:
            print(f"[P2PFL Web Services] Dropped batch ({len(entries)} entries): {e}")

    # --- HTTP helpers (sync, for immediate operations) ---

    def _get(self, path: str, *, timeout: int = 5) -> dict:
        response = self._client.get(self._base_url + path, timeout=timeout)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    def _post(self, path: str, data: Any, *, timeout: int = 5) -> dict:
        response = self._client.post(self._base_url + path, json=data, timeout=timeout)
        response.raise_for_status()
        if response.status_code == 204:
            return {}
        return response.json()  # type: ignore[no-any-return]

    def _patch(self, path: str, data: Any, *, timeout: int = 5) -> None:
        response = self._client.patch(self._base_url + path, json=data, timeout=timeout)
        response.raise_for_status()

    def _delete(self, path: str, *, timeout: int = 5) -> None:
        response = self._client.delete(self._base_url + path, timeout=timeout)
        response.raise_for_status()

    # --- Nodes ---

    def register_node(self, node: str, metadata: dict[str, Any] | None = None) -> None:
        """
        Register a node.

        Args:
            node: The node address.
            metadata: Optional system metadata (OS, hardware, GPU, geolocation).

        """
        try:
            data: dict[str, Any] = {"address": node}
            if metadata is not None:
                data["metadata"] = metadata
            result = self._post("/nodes", data)
            self.node_id[node] = result["id"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 409:
                # Node already registered — fetch its ID
                result = self._get(f"/nodes/{node}")
                self.node_id[node] = result["id"]
            else:
                print(f"[P2PFL Web Services] Failed to register node '{node}': {e}")
                print(f"  Check that '{self._base_url}' is valid (P2PFL_WEB_LOGGER_URL or ~/.p2pfl_env)")
                raise
        except httpx.HTTPError as e:
            print(f"[P2PFL Web Services] Failed to register node '{node}': {e}")
            print(f"  Check that '{self._base_url}' is valid (P2PFL_WEB_LOGGER_URL or ~/.p2pfl_env)")
            raise

    def update_node_status(self, node: str, state: str) -> None:
        """
        Update a node's lifecycle state.

        Args:
            node: The node address.
            state: The new state ("idle", "learning", "finished", "failed", "cancelled").

        """
        try:
            self._patch(f"/nodes/{node}/status", {"state": state})
        except httpx.HTTPError as e:
            print(f"[P2PFL Web Services] Failed to update node status for '{node}': {e}")

    def unregister_node(self, node: str) -> None:
        """
        Unregister a node.

        Args:
            node: The node address.

        """
        try:
            self._delete(f"/nodes/{node}")
        except httpx.HTTPError as e:
            print(f"[P2PFL Web Services] Failed to unregister node '{node}': {e}")

    # --- Experiments ---

    def create_experiment(self, node_address: str, **data: Any) -> int:
        """
        Create an experiment on the web services.

        ``node_address`` is separate because it is not part of the Experiment
        object — it identifies which node is joining the experiment (sent as
        a query parameter).  ``**data`` should come from
        ``experiment.to_dict()``.

        Args:
            node_address: The node participating in the experiment.
            **data: Experiment fields (exp_name, total_rounds, workflow, …).

        Returns:
            The server-assigned experiment ID.

        """
        exp_name: str = data.get("exp_name", "")
        self._node_exp[node_address] = exp_name

        # Always POST (sync, not batched via WS) — the node needs the experiment
        # ID before it can proceed. The backend is idempotent on exp_name.
        result = self._post(f"/experiments?node_address={node_address}", data)
        exp_id = result["id"]
        self._exp_id[exp_name] = exp_id
        return exp_id

    def update_experiment(self, exp_name: str, node_address: str, **changes: Any) -> None:
        """
        Buffer an experiment state update.

        Args:
            exp_name: Experiment name (for ID lookup).
            node_address: The node reporting the change.
            **changes: Fields that changed (round, status, current_stage, …).

        """
        exp_id = self._exp_id.get(exp_name)
        if exp_id is None:
            return
        self._enqueue({"type": "experiment_update", "experiment_id": exp_id, "node_address": node_address, **changes})

    # --- Batched data methods ---

    def send_log(self, time: datetime.datetime, node: str, level: int | str, message: str) -> None:
        """
        Buffer a log message.

        Args:
            time: The time of the message.
            node: The node address.
            level: The log level.
            message: The message.

        """
        self._enqueue(
            {
                "type": "log",
                "timestamp": time.isoformat() if isinstance(time, datetime.datetime) else str(time),
                "node_address": node,
                "level": str(level),
                "message": message,
            }
        )

    def send_local_metric(self, exp: str, round: int, metric: str, node: str, value: float, step: int) -> None:
        """
        Buffer a local metric.

        Args:
            exp: The experiment name.
            round: The round.
            metric: The metric name.
            node: The node address.
            value: The metric value.
            step: The training step.

        """
        exp_id = self._exp_id.get(exp)
        if exp_id is None:
            return
        self._enqueue(
            {
                "type": "metric",
                "experiment_id": exp_id,
                "node_address": node,
                "metric_name": metric,
                "round": round,
                "step": step,
                "value": value,
                "metric_type": "local",
            }
        )

    def send_global_metric(self, exp: str, round: int, metric: str, node: str, value: float) -> None:
        """
        Buffer a global metric.

        Args:
            exp: The experiment name.
            round: The round.
            metric: The metric name.
            node: The node address.
            value: The metric value.

        """
        exp_id = self._exp_id.get(exp)
        if exp_id is None:
            return
        self._enqueue(
            {
                "type": "metric",
                "experiment_id": exp_id,
                "node_address": node,
                "metric_name": metric,
                "round": round,
                "value": value,
                "metric_type": "global",
            }
        )

    def send_system_metric(self, node: str, metric: str, value: float, time: datetime.datetime) -> None:
        """
        Buffer a system metric.

        Args:
            node: The node address.
            metric: The metric name.
            value: The value.
            time: The timestamp.

        """
        self._enqueue(
            {
                "type": "system_metric",
                "node_address": node,
                "timestamp": time.isoformat(),
                "metric": metric,
                "value": value,
            }
        )

    def send_communication_log(
        self,
        node: str,
        timestamp: datetime.datetime,
        direction: str,
        cmd: str,
        source_dest: str,
        package_type: str,
        package_size: int,
        round_num: int | None = None,
        additional_info: dict | None = None,
    ) -> None:
        """
        Buffer a communication log.

        Args:
            node: The node address.
            timestamp: The timestamp of the communication.
            direction: Direction of communication ("sent" or "received").
            cmd: The command or message type.
            source_dest: Source (if receiving) or destination (if sending) node.
            package_type: Type of package ("message" or "weights").
            package_size: Size of the package in bytes.
            round_num: The federated learning round number.
            additional_info: Additional information as a dictionary.

        """
        # Skip heartbeat messages — they flood the buffer with no diagnostic value
        if cmd == "beat":
            return
        exp_name = self._node_exp.get(node)
        exp_id = self._exp_id.get(exp_name) if exp_name else None
        if exp_id is None:
            return
        self._enqueue(
            {
                "type": "message",
                "experiment_id": exp_id,
                "node_address": node,
                "cmd": cmd,
                "direction": direction,
                "peer": source_dest,
                "round": round_num,
                "size_bytes": package_size,
                "metadata": {"package_type": package_type, **(additional_info or {})},
            }
        )

    def get_pending_actions(self) -> list[dict]:
        """Get pending actions from the web services."""
        raise NotImplementedError
