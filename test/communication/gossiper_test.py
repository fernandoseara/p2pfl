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
"""Tests for the Gossiper class — unit tests and integration (propagation) tests."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from p2pfl.communication.commands.command import Command
from p2pfl.communication.protocols.protobuff.gossiper import Gossiper
from p2pfl.communication.protocols.protobuff.memory import MemoryCommunicationProtocol
from p2pfl.communication.protocols.protobuff.proto import node_pb2
from p2pfl.settings import Settings


def _make_gossiper(period: float = 0.05, messages_per_period: int = 100) -> Gossiper:
    """Create a Gossiper with mocked neighbors and a short period for fast tests."""
    neighbors = MagicMock()
    neighbors.get_all.return_value = {}
    build_msg = MagicMock()
    g = Gossiper(neighbors, build_msg, period=period, messages_per_period=messages_per_period)
    g.set_address("test_node")
    return g


def _make_msg(source: str = "other_node", hash_val: int = 42) -> node_pb2.RootMessage:
    """Build a minimal RootMessage with a gossip hash."""
    msg = node_pb2.RootMessage()
    msg.source = source
    msg.gossip_message.hash = hash_val
    return msg


def _make_client(addr: str) -> MagicMock:
    """Build a mock ProtobuffClient."""
    client = AsyncMock()
    client.nei_addr = addr
    client.is_connected.return_value = True
    client.has_temporal_connection.return_value = False
    return client


###
# _run: partial target splitting
###


@pytest.mark.asyncio
async def test_run_splits_large_target_list_across_periods():
    """Split large target lists across multiple gossip periods."""
    g = _make_gossiper(period=0.02, messages_per_period=2)

    clients = [_make_client(f"n{i}") for i in range(5)]
    msg = _make_msg()

    # Manually inject pending message (bypassing add_message's neighbor lookup)
    g._pending_msgs.append((msg, clients))

    # Run one gossip period manually
    await g.start()
    # Let it run long enough for multiple periods to drain the queue
    await asyncio.sleep(0.15)
    await g.stop()

    # All 5 clients should have been sent the message
    total_sends = sum(c.send.call_count for c in clients)
    assert total_sends == 5, f"Expected 5 sends total, got {total_sends}"


@pytest.mark.asyncio
async def test_run_partial_split_leaves_remainder():
    """After one period with budget=2 and 5 targets, 3 must remain pending."""
    g = _make_gossiper(period=100, messages_per_period=2)  # long period so we control timing

    clients = [_make_client(f"n{i}") for i in range(5)]
    msg = _make_msg()
    g._pending_msgs.append((msg, list(clients)))

    # Let _run execute one iteration then stop
    g._terminate_event.clear()

    async def stop_after_one_iteration():
        await asyncio.sleep(0.01)
        g._terminate_event.set()

    asyncio.create_task(stop_after_one_iteration())
    await g._run()

    # Budget was 2, so only 2 of the 5 clients should have been sent to
    total_sends = sum(c.send.call_count for c in clients)
    assert total_sends == 2
    # 3 remain pending
    assert len(g._pending_msgs) == 1
    assert len(g._pending_msgs[0][1]) == 3


###
# _run: send failure handling
###


@pytest.mark.asyncio
async def test_run_handles_send_failure_gracefully():
    """When a client.send raises, the gossiper logs a warning and continues."""
    g = _make_gossiper(period=0.02, messages_per_period=10)

    good_client = _make_client("good")
    bad_client = _make_client("bad")
    bad_client.send.side_effect = ConnectionError("gone")

    msg = _make_msg()
    g._pending_msgs.append((msg, [bad_client, good_client]))

    await g.start()
    await asyncio.sleep(0.1)
    await g.stop()

    # bad_client was attempted
    bad_client.send.assert_called_once()
    # good_client still got sent despite the earlier failure
    good_client.send.assert_called_once()


###
# gossip_weights
###


@pytest.mark.asyncio
async def test_gossip_weights_stops_on_early_stopping():
    """gossip_weights returns immediately when early_stopping_fn returns True."""
    g = _make_gossiper()
    g._neighbors.get_all.return_value = {}

    model_fn = MagicMock(return_value=(None, "cmd", 0, []))

    await g.gossip_weights(
        early_stopping_fn=lambda: True,
        get_candidates_fn=lambda: ["n1"],
        status_fn=lambda: "status",
        model_fn=model_fn,
        period=0.01,
        temporal_connection=False,
    )
    model_fn.assert_not_called()


@pytest.mark.asyncio
async def test_gossip_weights_stops_on_empty_candidates():
    """gossip_weights returns when get_candidates_fn returns an empty list."""
    g = _make_gossiper()

    call_count = 0

    def candidates():
        nonlocal call_count
        call_count += 1
        return []

    await g.gossip_weights(
        early_stopping_fn=lambda: False,
        get_candidates_fn=candidates,
        status_fn=lambda: "s",
        model_fn=lambda addr: (None, "cmd", 0, []),
        period=0.01,
        temporal_connection=False,
    )
    assert call_count == 1


@pytest.mark.asyncio
async def test_gossip_weights_sends_model_to_candidates():
    """gossip_weights sends models to sampled candidates."""
    g = _make_gossiper()

    client_a = _make_client("nodeA")
    g._neighbors.get_all.return_value = {
        "nodeA": (client_a, 1.0),
    }

    rounds = 0

    def candidates():
        nonlocal rounds
        rounds += 1
        if rounds <= 1:
            return ["nodeA"]
        return []  # stop after one round

    fake_model = MagicMock()

    await g.gossip_weights(
        early_stopping_fn=lambda: False,
        get_candidates_fn=candidates,
        status_fn=lambda: f"round-{rounds}",
        model_fn=lambda addr: (fake_model, "aggregate", 1, ["h1"]),
        period=0.01,
        temporal_connection=True,
    )
    client_a.send.assert_called_once_with(fake_model, temporal_connection=True)


@pytest.mark.asyncio
async def test_gossip_weights_skips_none_model():
    """When model_fn returns None model, the send is skipped."""
    g = _make_gossiper()

    client_a = _make_client("nodeA")
    g._neighbors.get_all.return_value = {"nodeA": (client_a, 1.0)}

    rounds = 0

    def candidates():
        nonlocal rounds
        rounds += 1
        return ["nodeA"] if rounds <= 1 else []

    await g.gossip_weights(
        early_stopping_fn=lambda: False,
        get_candidates_fn=candidates,
        status_fn=lambda: "s",
        model_fn=lambda addr: (None, "cmd", 0, []),
        period=0.01,
        temporal_connection=False,
    )
    client_a.send.assert_not_called()


@pytest.mark.asyncio
async def test_gossip_weights_exits_on_repeated_status():
    """gossip_weights exits when EXIT_ON_X_EQUAL_ROUNDS consecutive identical statuses are seen."""
    g = _make_gossiper()
    g._neighbors.get_all.return_value = {}

    Settings.gossip.EXIT_ON_X_EQUAL_ROUNDS = 3
    rounds = 0

    def candidates():
        nonlocal rounds
        rounds += 1
        return ["nodeA"]

    await asyncio.wait_for(
        g.gossip_weights(
            early_stopping_fn=lambda: False,
            get_candidates_fn=candidates,
            status_fn=lambda: "same_status",  # always identical
            model_fn=lambda addr: (None, "cmd", 0, []),
            period=0.01,
            temporal_connection=False,
        ),
        timeout=2.0,
    )

    # Must have run at least EXIT_ON_X_EQUAL_ROUNDS rounds before stopping
    assert rounds >= 3


###
# Gossiper unit tests (check_and_set_processed, add_message, circular buffer)
###


@pytest.fixture
def gossiper():
    """Gossiper instance with mock neighbors."""
    mock_neighbors = MagicMock()
    g = Gossiper(mock_neighbors, MagicMock())
    g.set_address("127.0.0.1:8000")
    return g


class TestGossiper:
    """Tests for Gossiper."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "msg_source,expected",
        [
            ("127.0.0.1:8000", False),  # Own message
            ("127.0.0.1:9000", True),  # New message from other
        ],
    )
    async def test_check_and_set_processed(self, gossiper, msg_source, expected):
        """Test message processing: own messages rejected, new messages accepted."""
        mock_msg = MagicMock()
        mock_msg.source = msg_source
        mock_msg.gossip_message.hash = 12345

        result = await gossiper.check_and_set_processed(mock_msg)
        assert result is expected

    @pytest.mark.asyncio
    async def test_check_and_set_processed_returns_false_for_duplicate(self, gossiper):
        """Test that duplicate messages return False."""
        mock_msg = MagicMock()
        mock_msg.source = "127.0.0.1:9000"
        mock_msg.gossip_message.hash = 12345

        assert await gossiper.check_and_set_processed(mock_msg) is True
        assert await gossiper.check_and_set_processed(mock_msg) is False

    @pytest.mark.asyncio
    async def test_add_message_queues_for_neighbors(self, gossiper):
        """Test that add_message queues message for all direct neighbors."""
        mock_client = MagicMock()
        gossiper._neighbors.get_all.return_value = {
            "127.0.0.1:9000": (mock_client, 0),
            "127.0.0.1:9001": (mock_client, 0),
        }

        mock_msg = MagicMock()
        mock_msg.source = "127.0.0.1:7000"

        await gossiper.add_message(mock_msg)

        assert len(gossiper._pending_msgs) == 1
        assert gossiper._pending_msgs[0][0] == mock_msg

    @pytest.mark.asyncio
    async def test_circular_buffer_limits_processed_messages(self, gossiper):
        """Test that processed messages list is limited to configured size."""
        limit = Settings.gossip.AMOUNT_LAST_MESSAGES_SAVED
        for i in range(limit + 5):
            mock_msg = MagicMock()
            mock_msg.source = "127.0.0.1:9000"
            mock_msg.gossip_message.hash = i

            await gossiper.check_and_set_processed(mock_msg)

        # The list should have removed old entries
        assert len(gossiper._processed_messages) <= limit + 1


###
# Integration tests — gossip propagation in star topology
###


class _TrackingCommand(Command):
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


async def _build_star(n_spokes: int) -> tuple[MemoryCommunicationProtocol, list[MemoryCommunicationProtocol], list[_TrackingCommand]]:
    """Build a star topology with 1 hub and n_spokes spokes, each with a _TrackingCommand."""
    commands: list[_TrackingCommand] = []

    # Hub
    hub_cmd = _TrackingCommand()
    commands.append(hub_cmd)
    hub = MemoryCommunicationProtocol(commands=[hub_cmd])
    hub.set_address("hub")
    await hub.start()

    # Spokes
    spokes: list[MemoryCommunicationProtocol] = []
    for i in range(n_spokes):
        cmd = _TrackingCommand()
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
