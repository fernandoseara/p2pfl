#
# This file is part of the p2pfl (see https://github.com/pguijas/p2pfl).
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
"""Observer that bridges Experiment attribute changes to the logger."""

from __future__ import annotations

from typing import Any

from p2pfl.management.logger import logger
from p2pfl.workflow.engine.observable import Observer


class ExperimentLoggerObserver(Observer):
    """
    Bridge Experiment state changes to the logger singleton.

    Args:
        address: The node address to associate with log events.

    """

    def __init__(self, address: str) -> None:
        """Initialize with the node address."""
        self._address = address

    def update(self, field_name: str, value: Any) -> None:
        """Forward attribute change to ``logger.on_experiment_change``."""
        logger.on_experiment_change(self._address, field_name, value)
