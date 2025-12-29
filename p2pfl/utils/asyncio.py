#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
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

"""Asynchronous utility functions."""

import asyncio
from functools import wraps


def run_async(coro):
    """
    Run an asynchronous coroutine in a blocking manner.

    This function checks if an event loop is already running. If it is, it ensures the coroutine
    is run in the current loop. If not, it creates a new event loop and runs the coroutine.

    Args:
        coro: The coroutine to run.

    Returns:
        The result of the coroutine.

    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    if loop.is_running():
        future = asyncio.ensure_future(coro)
        done, _ = loop.run_until_complete(asyncio.wait([future]))
        return done.pop().result()
    else:
        return loop.run_until_complete(coro)

def sync_or_async(async_func):
    """Support both sync and async calling of a method."""

    @wraps(async_func)
    def wrapper(self, *args, **kwargs):
        if asyncio.current_task():
            # Called from async context
            return async_func(self, *args, **kwargs)  # Call original async method
        else:
            # Called from sync context
            return run_async(async_func(self, *args, **kwargs))
    return wrapper
