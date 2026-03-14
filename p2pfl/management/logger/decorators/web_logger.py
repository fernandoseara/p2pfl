#
# This file is part of the p2pfl distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2026 Pedro Guijas Bravo.
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

"""Web Logger."""

from __future__ import annotations

import datetime
import logging
import os
from typing import TYPE_CHECKING, Any

from p2pfl.management.logger.decorators.logger_decorator import LoggerDecorator
from p2pfl.management.logger.logger import P2PFLogger
from p2pfl.management.p2pfl_web_services import P2pflWebServices

if TYPE_CHECKING:
    from p2pfl.workflow.engine.experiment import Experiment

#########################################
#    Logging handler (transmit logs)    #
#########################################


class DictFormatter(logging.Formatter):
    """Formatter (logging) that returns a dictionary with the log record attributes."""

    def format(self, record):
        """
        Format the log record as a dictionary.

        Args:
            record: The log record.

        """
        # Get node
        if not hasattr(record, "node"):
            raise ValueError("The log record must have a 'node' attribute.")
        log_dict = {
            "timestamp": datetime.datetime.fromtimestamp(record.created),
            "level": record.levelname,
            "node": record.node,  # type: ignore
            "message": record.getMessage(),
        }
        return log_dict


class P2pflWebLogHandler(logging.Handler):
    """
    Custom logging handler that sends log entries to the API.

    Args:
        p2pfl_web: The P2PFL Web Services.

    """

    def __init__(self, p2pfl_web: P2pflWebServices):
        """Initialize the handler."""
        super().__init__()
        self.p2pfl_web = p2pfl_web
        self.formatter = DictFormatter()  # Instantiate the custom formatter

    def emit(self, record):
        """
        Emit the log record.

        Args:
            record: The log record.

        """
        # Format the log record using the custom formatter
        log_message = self.formatter.format(record)  # type: ignore
        # Send log entry to the API
        self.p2pfl_web.send_log(
            log_message["timestamp"],  # type: ignore
            log_message["node"],  # type: ignore
            log_message["level"],  # type: ignore
            log_message["message"],  # type: ignore
        )


class WebP2PFLogger(LoggerDecorator):
    """Web logger decorator."""

    def __init__(self, p2pflogger: P2PFLogger):
        """Initialize the logger."""
        super().__init__(p2pflogger)
        self._p2pfl_web_services: P2pflWebServices | None = None
        self._ended_experiments: set[str] = set()
        self._node_callbacks: dict[str, Any] = {}

        # Load credentials from .p2pfl_env file if it exists
        self._load_env_file()

        # Try to auto-connect using environment variables
        self.connect()

    def _load_env_file(self) -> None:
        """Load environment variables from ~/.p2pfl_env if it exists."""
        # Skip loading in test mode to avoid interference with tests
        if os.environ.get("P2PFL_TESTING", "").lower() in ("1", "true"):
            return

        # Skip if environment variables are already set
        if "P2PFL_WEB_LOGGER_URL" in os.environ and "P2PFL_WEB_LOGGER_KEY" in os.environ:
            return

        env_file = os.path.join(os.path.expanduser("~"), ".p2pfl_env")
        if os.path.exists(env_file):
            try:
                with open(env_file) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, value = line.split("=", 1)
                            # Only set if not already in environment
                            if key not in os.environ:
                                os.environ[key] = value
                super().debug("WebP2PFLogger", f"Loaded credentials from {env_file}")
            except Exception as e:
                super().warning("WebP2PFLogger", f"Could not load {env_file}: {e}")

    def connect(
        self,
        p2pfl_web_url: str | None = None,
        p2pfl_web_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Establish connection to web services.

        Args:
            p2pfl_web_url: The URL of the web services (or P2PFL_WEB_LOGGER_URL env var)
            p2pfl_web_key: The API key (or P2PFL_WEB_LOGGER_KEY env var)
            **kwargs: Additional parameters (for compatibility)

        """
        # Get parameters from function args or environment variables
        url = p2pfl_web_url or os.environ.get("P2PFL_WEB_LOGGER_URL")
        key = p2pfl_web_key or os.environ.get("P2PFL_WEB_LOGGER_KEY")

        # Check if we have the required parameters
        if url is None or key is None:
            if url is not None or key is not None:
                super().warning("WebP2PFLogger", "P2PFL Web URL or key provided but incomplete. Both URL and key are required.")
            return

        # If already connected, skip
        if self._p2pfl_web_services is not None:
            super().debug("WebP2PFLogger", "Web services already connected, skipping re-initialization")
            return

        # Connect to web services
        try:
            self._p2pfl_web_services = P2pflWebServices(str(url), str(key))
            self.add_handler(P2pflWebLogHandler(self._p2pfl_web_services))
            super().debug("WebP2PFLogger", f"Successfully connected to P2PFL Web Services at {url}")
        except Exception as e:
            super().warning("WebP2PFLogger", f"Failed to connect to P2PFL Web Services: {e}")
            self._p2pfl_web_services = None

    def on_experiment_change(self, address: str, field_name: str, value: Any) -> None:
        """Forward experiment changes to web services."""
        super().on_experiment_change(address, field_name, value)
        if self._p2pfl_web_services is not None:
            try:
                node_data = self.get_nodes().get(address)
                if node_data is None:
                    return
                experiment: Experiment | None = node_data.get("Experiment")
                if experiment is None:
                    return
                self._p2pfl_web_services.update_experiment(experiment.exp_name, address, **{field_name: value})
            except Exception as e:
                super().warning("WebP2PFLogger", f"Error forwarding {field_name} update: {e}")

    def experiment_started(self, node: str, experiment: Experiment) -> None:
        """
        Handle experiment start for web services.

        Args:
            node: The node address.
            experiment: The experiment object containing metadata.

        """
        if self._p2pfl_web_services is not None:
            try:
                self._p2pfl_web_services.create_experiment(node, **experiment.to_dict(exclude_none=True))
                super().debug("WebP2PFLogger", f"Experiment '{experiment.exp_name}' created for node {node}")
            except Exception as e:
                super().warning("WebP2PFLogger", f"Failed to create experiment on web services: {e}")

        # Call parent's experiment_started
        super().experiment_started(node, experiment)

    def experiment_ended(self, address: str, experiment: Experiment, status: str) -> None:
        """Send terminal status and flush buffered data on experiment end."""
        if self._p2pfl_web_services is not None and experiment.exp_name not in self._ended_experiments:
            self._ended_experiments.add(experiment.exp_name)
            try:
                self._p2pfl_web_services.update_experiment(experiment.exp_name, address, status=status)
                self._p2pfl_web_services.flush()
            except Exception as e:
                super().warning("WebP2PFLogger", f"Error flushing on experiment end: {e}")
        super().experiment_ended(address, experiment, status)

    def log_metric(self, addr: str, metric: str, value: float, step: int | None = None, round: int | None = None) -> None:
        """
        Log a metric.

        Args:
            addr: The node name.
            metric: The metric to log.
            value: The value.
            step: The step.
            round: The round.

        """
        super().log_metric(addr=addr, metric=metric, value=value, step=step, round=round)

        if self._p2pfl_web_services is not None:
            # Get Experiment and round
            try:
                nodes = self.get_nodes()
                experiment: Experiment = nodes[addr]["Experiment"]
                current_round = nodes[addr].get("round", 0)
            except KeyError:
                # If no experiment is registered for this node, skip web logging
                return

            if step is None:
                # Global Metrics
                self._p2pfl_web_services.send_global_metric(experiment.exp_name, current_round, metric, addr, value)
            else:
                # Local Metrics
                self._p2pfl_web_services.send_local_metric(experiment.exp_name, current_round, metric, addr, value, step)

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
        """
        Log a communication event and send it to web services if connected.

        Args:
            node: The node address.
            direction: Direction of communication ("sent" or "received").
            cmd: The command or message type.
            source_dest: Source (if receiving) or destination (if sending) node.
            package_type: Type of package ("message" or "weights").
            package_size: Size of the package in bytes (if available).
            round_num: The federated learning round number (if applicable).
            additional_info: Additional information as a dictionary.

        """
        # Call parent's method first
        super().log_communication(
            node=node,
            direction=direction,
            cmd=cmd,
            source_dest=source_dest,
            package_type=package_type,
            package_size=package_size,
            round_num=round_num,
            additional_info=additional_info,
        )

        # Send to web services if connected
        if self._p2pfl_web_services is not None:
            # Create timestamp
            now = datetime.datetime.now()

            # Send as a structured communication log
            try:
                self._p2pfl_web_services.send_communication_log(
                    node=node,
                    timestamp=now,
                    direction=direction,
                    cmd=cmd,
                    source_dest=source_dest,
                    package_type=package_type,
                    package_size=package_size,
                    round_num=round_num,
                    additional_info=additional_info,
                )
            except Exception as e:
                # Error handling
                super().warning("WebP2PFLogger", f"Error sending communication log to web services: {e}")

    def register_node(self, node: str) -> None:
        """
        Register a node.

        Args:
            node: The node address.

        """
        super().register_node(node)
        if self._p2pfl_web_services is not None:
            from p2pfl.management.node_monitor import collect_node_metadata

            metadata = collect_node_metadata()
            self._p2pfl_web_services.register_node(node, metadata=metadata)

            # Register a monitor callback to push system metrics in real-time
            ws = self._p2pfl_web_services

            def _push_metrics(ts: datetime.datetime, metrics: dict[str, float], _node: str = node) -> None:
                for metric_name, value in metrics.items():
                    ws.send_system_metric(_node, metric_name, value, ts)

            self._node_callbacks[node] = _push_metrics
            self.node_monitor.add_callback(_push_metrics)

    def unregister_node(self, node: str) -> None:
        """
        Unregister a node.

        Args:
            node: The node address.

        """
        cb = self._node_callbacks.pop(node, None)
        if cb is not None:
            self.node_monitor.remove_callback(cb)
        super().unregister_node(node)
        if self._p2pfl_web_services is not None:
            self._p2pfl_web_services.unregister_node(node)

    def finish(self) -> None:
        """
        Finish the current experiment for web services.

        Flushes any buffered data before finishing.
        The connection remains alive for potential future experiments.
        """
        if self._p2pfl_web_services is not None:
            self._p2pfl_web_services.flush()
        # Call parent's finish
        super().finish()

    def reset(self) -> None:
        """Reset state between experiments."""
        self._ended_experiments.clear()
        super().reset()

    def cleanup(self) -> None:
        """Cleanup: stop the flush task and send remaining data."""
        if self._p2pfl_web_services is not None:
            self._p2pfl_web_services.stop()
        super().cleanup()
