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
"""Tests for gossip message propagation and dedup in star topology."""

import asyncio

import pytest

from p2pfl.communication.commands.command import Command
from p2pfl.communication.protocols.protobuff.memory import MemoryCommunicationProtocol
from p2pfl.settings import Settings
from p2pfl.utils.utils import set_standalone_settings

set_standalone_settings()


class TrackingCommand(Command):
    """Command that records every invocation with source and args."""

    def __init__(self) -> None:
        """Initialize the tracking command."""
        self.received: list[tuple[str, tuple]] = []

    @staticmethod
    def get_name() -> str:
        """Get the name of the command."""
        return "test_gossip_msg"

    async def execute(self, source: str, round: int, *args, **kwargs) -> None:
        """Record the message."""
        self.received.append((source, args))


async def _build_star(n_spokes: int) -> tuple[MemoryCommunicationProtocol, list[MemoryCommunicationProtocol], list[TrackingCommand]]:
    """Build a star topology with 1 hub and n_spokes spokes, each with a TrackingCommand."""
    commands: list[TrackingCommand] = []

    # Hub
    hub_cmd = TrackingCommand()
    commands.append(hub_cmd)
    hub = MemoryCommunicationProtocol(commands=[hub_cmd])
    hub.set_address("hub")
    await hub.start()

    # Spokes
    spokes: list[MemoryCommunicationProtocol] = []
    for i in range(n_spokes):
        cmd = TrackingCommand()
        commands.append(cmd)
        spoke = MemoryCommunicationProtocol(commands=[cmd])
        spoke.set_address(f"spoke_{i}")
        await spoke.start()
        await hub.connect(spoke.get_address())
        spokes.append(spoke)

    # Wait for neighbor discovery to settle
    await asyncio.sleep(Settings.gossip.PERIOD * 2)

    return hub, spokes, commands


async def _stop_all(hub, spokes):
    """Stop all protocols."""
    for s in spokes:
        await s.stop()
    await hub.stop()


@pytest.mark.asyncio
async def test_spoke_to_spoke_gossip_delivery():
    """A gossip message from spoke_0 must reach the hub and all other spokes exactly once."""
    hub, spokes, commands = await _build_star(4)
    hub_cmd, *spoke_cmds = commands

    try:
        # spoke_0 broadcasts a gossip message
        msg = spokes[0].build_msg("test_gossip_msg", args=["hello", "world"])
        await spokes[0].broadcast_gossip(msg)

        # Wait for gossip forwarding (hub receives instantly, forwards in next period)
        await asyncio.sleep(Settings.gossip.PERIOD * 3)

        # Hub must have received the message
        assert len(hub_cmd.received) == 1, f"Hub received {len(hub_cmd.received)} messages, expected 1"
        assert hub_cmd.received[0] == (spokes[0].get_address(), ("hello", "world"))

        # spoke_0 (originator) must NOT receive its own message
        assert len(spoke_cmds[0].received) == 0, f"Originator received {len(spoke_cmds[0].received)} messages, expected 0"

        # spoke_1, spoke_2, spoke_3 must each receive exactly 1 message
        for i in range(1, 4):
            assert len(spoke_cmds[i].received) == 1, f"spoke_{i} received {len(spoke_cmds[i].received)} messages, expected 1"
            assert spoke_cmds[i].received[0] == (spokes[0].get_address(), ("hello", "world"))
    finally:
        await _stop_all(hub, spokes)


@pytest.mark.asyncio
async def test_hub_gossip_reaches_all_spokes():
    """A gossip message from the hub must reach all spokes exactly once."""
    hub, spokes, commands = await _build_star(4)
    hub_cmd, *spoke_cmds = commands

    try:
        msg = hub.build_msg("test_gossip_msg", args=["from_hub"])
        await hub.broadcast_gossip(msg)

        await asyncio.sleep(Settings.gossip.PERIOD * 3)

        # Hub must NOT receive its own message
        assert len(hub_cmd.received) == 0, f"Hub received {len(hub_cmd.received)} messages, expected 0"

        # All spokes must receive exactly 1 message
        for i in range(4):
            assert len(spoke_cmds[i].received) == 1, f"spoke_{i} received {len(spoke_cmds[i].received)} messages, expected 1"
            assert spoke_cmds[i].received[0] == (hub.get_address(), ("from_hub",))
    finally:
        await _stop_all(hub, spokes)


@pytest.mark.asyncio
async def test_multiple_spokes_broadcast_simultaneously():
    """All spokes broadcast at the same time; each spoke must receive messages from all other spokes exactly once."""
    hub, spokes, commands = await _build_star(4)
    hub_cmd, *spoke_cmds = commands

    try:
        # All spokes broadcast simultaneously
        for i, spoke in enumerate(spokes):
            msg = spoke.build_msg("test_gossip_msg", args=[f"from_spoke_{i}"])
            await spoke.broadcast_gossip(msg)

        # Wait for gossip forwarding
        await asyncio.sleep(Settings.gossip.PERIOD * 3)

        # Hub must have received 4 messages (one from each spoke)
        assert len(hub_cmd.received) == 4, f"Hub received {len(hub_cmd.received)} messages, expected 4"

        # Each spoke must receive exactly 3 messages (from the other 3 spokes, forwarded via hub)
        for i in range(4):
            assert len(spoke_cmds[i].received) == 3, f"spoke_{i} received {len(spoke_cmds[i].received)} messages, expected 3"
            # Must not include own message
            sources = [src for src, _ in spoke_cmds[i].received]
            assert spokes[i].get_address() not in sources, f"spoke_{i} received its own message"
    finally:
        await _stop_all(hub, spokes)


@pytest.mark.asyncio
async def test_gossip_dedup_prevents_duplicate_processing():
    """Sending the same message twice must not result in duplicate processing."""
    hub, spokes, commands = await _build_star(2)
    hub_cmd, *spoke_cmds = commands

    try:
        # spoke_0 sends the same message object twice
        msg = spokes[0].build_msg("test_gossip_msg", args=["dup_test"])
        await spokes[0].broadcast_gossip(msg)
        await spokes[0].broadcast_gossip(msg)  # same hash

        await asyncio.sleep(Settings.gossip.PERIOD * 3)

        # Hub and spoke_1 must receive exactly 1 message (second is deduped)
        assert len(hub_cmd.received) == 1, f"Hub received {len(hub_cmd.received)} messages, expected 1"
        assert len(spoke_cmds[1].received) == 1, f"spoke_1 received {len(spoke_cmds[1].received)} messages, expected 1"
    finally:
        await _stop_all(hub, spokes)
