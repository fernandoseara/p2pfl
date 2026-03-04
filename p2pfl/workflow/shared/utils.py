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
"""Shared async utilities for workflow stages."""

from __future__ import annotations

import asyncio

from p2pfl.management.logger import logger


async def wait_with_timeout(event: asyncio.Event, timeout: float, address: str, msg: str) -> bool:
    """
    Clear an event, wait for it with a timeout, and log a warning on timeout.

    Args:
        event: The asyncio event to wait on.
        timeout: Timeout in seconds.
        address: Node address for logging.
        msg: Warning message to log on timeout.

    Returns:
        True if the event was set before the timeout, False otherwise.

    """
    event.clear()
    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
        return True
    except TimeoutError:
        logger.warning(address, msg)
        return False
