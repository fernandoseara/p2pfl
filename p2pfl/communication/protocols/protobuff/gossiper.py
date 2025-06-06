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

"""Protocol agnostic gossiper."""

import asyncio
import random
from typing import Any, Callable, List, Optional

from p2pfl.communication.protocols.protobuff.client import ProtobuffClient
from p2pfl.communication.protocols.protobuff.neighbors import Neighbors
from p2pfl.communication.protocols.protobuff.proto import node_pb2
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.utils.node_component import NodeComponent


class Gossiper(NodeComponent):
    """
    Async-compatible Gossiper that periodically spreads messages and models 
    to selected neighbors using protocol-agnostic logic.
    
    """

    def __init__(
        self,
        neighbors: Neighbors,
        period: Optional[float] = None,
        messages_per_period: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._neighbors = neighbors
        self.period = period or Settings.gossip.PERIOD
        self.messages_per_period = messages_per_period or Settings.gossip.MESSAGES_PER_PERIOD

        # State
        self._processed_messages: List[int] = []
        self._pending_msgs: List[tuple[node_pb2.RootMessage, List[ProtobuffClient]]] = []

        # Concurrency
        self._processed_messages_lock = asyncio.Lock()
        self._pending_msgs_lock = asyncio.Lock()
        self._terminate_event = asyncio.Event()

        self._task: Optional[asyncio.Task] = None
        self.name = "gossiper-async"

    def set_addr(self, addr: str) -> str:
        """Assigns address and updates gossiper thread name."""
        addr = super().set_addr(addr)
        self.name = f"gossiper-async-{addr}"
        return addr

    async def start(self) -> None:
        """Launch the gossiping task."""
        logger.info(self.address, "🏁 Starting gossiper...")
        self._terminate_event.clear()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Signal the gossiper to stop and wait for completion."""
        logger.info(self.address, "🛑 Stopping gossiper...")
        self._terminate_event.set()
        if self._task:
            await self._task

    async def add_message(self, msg: node_pb2.RootMessage) -> None:
        """
        Queues a message to be gossiped to all direct neighbors, excluding the source and self.
        """
        async with self._pending_msgs_lock:
            neighbors = [
                v[0] for addr, v in self._neighbors.get_all(only_direct=True).items()
                if addr != self.address and addr != msg.source
            ]
            self._pending_msgs.append((msg, neighbors))

    async def check_and_set_processed(self, msg: node_pb2.RootMessage) -> bool:
        """
        Marks a message as processed if it hasn't been seen yet.
        Returns True if the message was new, False otherwise.
        """
        if msg.source == self.address:
            return False

        async with self._processed_messages_lock:
            if msg.message.hash in self._processed_messages:
                return False

            if len(self._processed_messages) >= Settings.gossip.AMOUNT_LAST_MESSAGES_SAVED:
                self._processed_messages.pop(0)

            self._processed_messages.append(msg.message.hash)
            return True

    async def _run(self) -> None:
        """Main loop that periodically sends pending messages to neighbors."""
        while not self._terminate_event.is_set():
            start_time = asyncio.get_event_loop().time()
            messages_to_send = []
            remaining = self.messages_per_period

            async with self._pending_msgs_lock:
                while remaining > 0 and self._pending_msgs:
                    msg, targets = self._pending_msgs[0]
                    if len(targets) <= remaining:
                        messages_to_send.append((msg, targets))
                        self._pending_msgs.pop(0)
                        remaining -= len(targets)
                    else:
                        messages_to_send.append((msg, targets[:remaining]))
                        self._pending_msgs[0] = (msg, targets[remaining:])
                        remaining = 0

            # Send messages
            for msg, clients in messages_to_send:
                for client in clients:
                    await client.send(msg)

            elapsed = asyncio.get_event_loop().time() - start_time
            await asyncio.sleep(max(0, self.period - elapsed))

    async def gossip_weights(
        self,
        early_stopping_fn: Callable[[], bool],
        get_candidates_fn: Callable[[], List[str]],
        status_fn: Callable[[], Any],
        model_fn: Callable[[str], Any],
        period: float,
        temporal_connection: bool,
    ) -> None:
        """
        Gossips model weights synchronously to a random subset of candidate nodes.

        Continues until early stopping is triggered or neighbor states stabilize.

        Args:
            early_stopping_fn: Whether to stop early.
            get_candidates_fn: Returns addresses of neighbor nodes.
            status_fn: Returns the current node state.
            model_fn: Generates the model to send to each node.
            period: Delay between gossip rounds.
            temporal_connection: Whether to use temporary connections.
        """
        last_status: List[Any] = []
        j = 0

        while True:
            start_time = asyncio.get_event_loop().time()

            if early_stopping_fn():
                logger.info(self.address, "Stopping model gossip process.")
                return

            candidates = get_candidates_fn()
            if not candidates:
                logger.info(self.address, "🤫 Gossip finished.")
                return

            logger.debug(self.address, f"👥 Remaining gossip targets: {candidates}")

            current_status = status_fn()
            if len(last_status) < Settings.gossip.EXIT_ON_X_EQUAL_ROUNDS:
                last_status.append(current_status)
            else:
                last_status[j] = str(current_status)
                j = (j + 1) % Settings.gossip.EXIT_ON_X_EQUAL_ROUNDS

                if all(status == last_status[0] for status in last_status):
                    logger.info(self.address, f"⏹️  Gossip stopped: {Settings.gossip.EXIT_ON_X_EQUAL_ROUNDS} identical rounds.")
                    logger.debug(self.address, f"Final status: {last_status[-1]}")
                    return

            # Randomly select a subset of candidate nodes
            sample_size = min(Settings.gossip.MODELS_PER_ROUND, len(candidates))
            sampled = random.sample(candidates, sample_size)

            clients = [
                v[0] for k, v in self._neighbors.get_all(only_direct=False).items()
                if k in sampled
            ]

            for client in clients:
                model = model_fn(client.nei_addr)
                if model is not None:
                    logger.debug(self.address, f"🗣️ Sending model to {client.nei_addr}")
                    await client.send(model, temporal_connection=temporal_connection)

            elapsed = asyncio.get_event_loop().time() - start_time
            await asyncio.sleep(max(0, period - elapsed))
