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

"""Protocol agnostic heartbeater."""

import asyncio
import contextlib
import time
from collections.abc import Callable

from p2pfl.communication.protocols.protobuff.neighbors import Neighbors
from p2pfl.communication.protocols.protobuff.proto import node_pb2
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.utils.node_component import NodeComponent

heartbeater_cmd_name = "beat"


class Heartbeater(NodeComponent):
    """Async Heartbeater for agnostic communication protocol."""

    def __init__(self, neighbors: Neighbors, build_msg: Callable[..., node_pb2.RootMessage]) -> None:
        """Initialize the heartbeat task."""
        self.__neighbors = neighbors
        self.__build_beat_message: Callable[[float], node_pb2.RootMessage] = lambda time: build_msg(heartbeater_cmd_name, content=[time])
        self.__heartbeat_task: asyncio.Task | None = None
        self.__stop_event = asyncio.Event()
        self.name = "heartbeater-async-unknown"

    def set_address(self, address: str) -> str:
        """Set the address and update the task name."""
        address = super().set_address(address)
        self.name = f"heartbeater-async-{address}"
        return address

    async def start(self, period: float | None = None, timeout: float | None = None) -> None:
        """Start the async heartbeat task."""
        self.__heartbeat_task = asyncio.create_task(self.__heartbeater(period, timeout))

    async def stop(self) -> None:
        """Stop the heartbeat task gracefully."""
        self.__stop_event.set()
        if self.__heartbeat_task:
            await self.__heartbeat_task

    async def beat(self, nei: str, time: float) -> None:
        """Update heartbeat time for a neighbor."""
        if nei == self.address:
            return
        await self.__neighbors.refresh_or_add(nei, time)

    async def __heartbeater(
        self,
        period: float | None = None,
        timeout: float | None = None,
    ) -> None:
        if period is None:
            period = Settings.heartbeat.PERIOD
        if timeout is None:
            timeout = Settings.heartbeat.TIMEOUT

        toggle = False

        while not self.__stop_event.is_set():
            t = time.time()

            # Check for timeouts every other loop
            if toggle:
                neis = self.__neighbors.get_all()
                for nei, (_, last_seen) in neis.items():
                    if t - last_seen > timeout:
                        logger.info(
                            self.address,
                            f"Heartbeat timeout for {nei} ({t - last_seen}). Removing...",
                        )
                        await self.__neighbors.remove(nei)
            else:
                toggle = True

            # Send heartbeat
            beat_msg = self.__build_beat_message(time.time())
            beat_msg.gossip_message.ttl = Settings.gossip.TTL
            for client, _ in self.__neighbors.get_all(only_direct=True).values():
                try:
                    await client.send(beat_msg, raise_error=False, disconnect_on_error=True)
                except Exception as e:
                    logger.warning(self.address, f"Failed to send heartbeat to {client}: {e}")

            # Sleep
            elapsed = time.time() - t
            sleep_time = max(0, period - elapsed)
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self.__stop_event.wait(), timeout=sleep_time)
