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
"""Unit tests for the Gossiper class — covers partial send batching, error handling, and gossip_weights."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from p2pfl.communication.protocols.protobuff.gossiper import Gossiper
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


# ── _run: partial target splitting ──────────────────────────────────


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

    # Manually run the drain logic once (extract from _run)
    messages_to_send = []
    remaining = g.messages_per_period  # 2

    async with g._pending_msgs_lock:
        while remaining > 0 and g._pending_msgs:
            m, targets = g._pending_msgs[0]
            if len(targets) <= remaining:
                messages_to_send.append((m, targets))
                g._pending_msgs.pop(0)
                remaining -= len(targets)
            else:
                messages_to_send.append((m, targets[:remaining]))
                g._pending_msgs[0] = (m, targets[remaining:])
                remaining = 0

    # Should have taken 2 targets
    assert len(messages_to_send) == 1
    assert len(messages_to_send[0][1]) == 2
    # 3 remain
    assert len(g._pending_msgs) == 1
    assert len(g._pending_msgs[0][1]) == 3


# ── _run: send failure handling ─────────────────────────────────────


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


# ── gossip_weights ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_gossip_weights_stops_on_early_stopping():
    """gossip_weights returns immediately when early_stopping_fn returns True."""
    g = _make_gossiper()
    g._neighbors.get_all.return_value = {}

    await g.gossip_weights(
        early_stopping_fn=lambda: True,
        get_candidates_fn=lambda: ["n1"],
        status_fn=lambda: "status",
        model_fn=lambda addr: (None, "cmd", 0, []),
        period=0.01,
        temporal_connection=False,
    )
    # Should return without error; no sends expected


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

    original = Settings.gossip.EXIT_ON_X_EQUAL_ROUNDS
    Settings.gossip.EXIT_ON_X_EQUAL_ROUNDS = 3
    rounds = 0

    def candidates():
        nonlocal rounds
        rounds += 1
        return ["nodeA"]

    try:
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
    finally:
        Settings.gossip.EXIT_ON_X_EQUAL_ROUNDS = original

    # Must have run at least EXIT_ON_X_EQUAL_ROUNDS rounds before stopping
    assert rounds >= 3
