#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/federated_learning_p2p).
# Copyright (c) 2022 Pedro Guijas Bravo.
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
"""Async logger."""

from __future__ import annotations

import atexit
import logging
import multiprocessing
import queue
import threading
import traceback
from logging.handlers import QueueHandler, QueueListener
from typing import Any

from p2pfl.management.logger.decorators.logger_decorator import LoggerDecorator
from p2pfl.management.logger.logger import P2PFLogger


class AsyncLogger(LoggerDecorator):
    """
    Async logger decorator.

    Dispatches high-volume fire-and-forget methods (log, log_metric,
    log_communication, on_experiment_change) to a lightweight queue + daemon
    threads so they never block the caller.

    Lifecycle methods (register_node, experiment_started, etc.) remain
    synchronous because they set up state that subsequent calls depend on.
    """

    _NUM_WORKERS = 2
    _POISON = None

    def __init__(self, p2pflogger: P2PFLogger) -> None:
        """Initialize the logger."""
        super().__init__(p2pflogger)

        # Lightweight queue + daemon threads for fire-and-forget logger operations
        self._queue: queue.SimpleQueue[tuple | None] = queue.SimpleQueue()
        self._workers: list[threading.Thread] = []
        for i in range(self._NUM_WORKERS):
            t = threading.Thread(target=self._worker_loop, name=f"p2pfl-log-worker-{i}", daemon=True)
            t.start()
            self._workers.append(t)

        # Set up asynchronous logging for Python log records
        self.log_queue: multiprocessing.Queue[logging.LogRecord] = multiprocessing.Queue()
        queue_handler = QueueHandler(self.log_queue)
        self._p2pfl_logger.add_handler(queue_handler)

        # Set up a listener for the queue
        self.queue_listener = QueueListener(self.log_queue)
        self.queue_listener.start()
        listener_thread = getattr(self.queue_listener, "_thread", None)
        if listener_thread is not None:
            listener_thread.name = "p2pfl-log-listener"

        # Name the mp.Queue feeder thread (spawned lazily; force-start it)
        start_thread = getattr(self.log_queue, "_start_thread", None)
        if start_thread is not None:
            start_thread()
        feeder_thread = getattr(self.log_queue, "_thread", None)
        if feeder_thread is not None:
            feeder_thread.name = "p2pfl-log-feeder"

        # Register cleanup function to close the queue on exit
        atexit.register(self.cleanup)

    def add_handler(self, handler: logging.Handler) -> None:
        """Add a handler to the logger."""
        self.queue_listener.handlers = self.queue_listener.handlers + (handler,)

    def _worker_loop(self) -> None:
        """Pull (fn, args, kwargs) tuples from the queue until a poison pill."""
        while True:
            item = self._queue.get()
            if item is self._POISON:
                return
            fn, args, kwargs = item
            try:
                fn(*args, **kwargs)
            except Exception:
                traceback.print_exc()

    # --- High-volume methods (dispatched to queue) ---

    def log(self, level: int, node: str, message: str) -> None:
        """Log a message (non-blocking)."""
        self._queue.put((super().log, (level, node, message), {}))

    def log_metric(self, addr: str, metric: str, value: float, step: int | None = None, round: int | None = None) -> None:
        """Log a metric (non-blocking)."""
        self._queue.put((super().log_metric, (), {"addr": addr, "metric": metric, "value": value, "step": step, "round": round}))

    def log_communication(
        self,
        node: str,
        direction: str,
        cmd: str,
        source_dest: str,
        package_type: str,
        package_size: int,
        round_num: int | None = None,
        additional_info: dict[str, Any] | None = None,
    ) -> None:
        """Log a communication event (non-blocking)."""
        self._queue.put(
            (
                super().log_communication,
                (),
                {
                    "node": node,
                    "direction": direction,
                    "cmd": cmd,
                    "source_dest": source_dest,
                    "package_type": package_type,
                    "package_size": package_size,
                    "round_num": round_num,
                    "additional_info": additional_info,
                },
            )
        )

    def on_experiment_change(self, address: str, field_name: str, value: Any) -> None:
        """Handle experiment attribute change (non-blocking)."""
        self._queue.put((super().on_experiment_change, (address, field_name, value), {}))

    # --- Cleanup ---

    def cleanup(self) -> None:
        """Cleanup the logger."""
        # Send poison pills to stop workers and wait for them to drain
        for _ in self._workers:
            self._queue.put(self._POISON)
        for w in self._workers:
            w.join()
        if self.queue_listener:
            self.queue_listener.stop()
        self.log_queue.close()
        super().cleanup()
