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
"""Model transfer gating utilities for workflow stages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from p2pfl.management.logger import logger

if TYPE_CHECKING:
    from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol


class ModelGate:
    """
    Gate that negotiates model transfers between peers.

    The sender asks the receiver whether it wants a model via a
    stage-specific pre-send command, and only transmits the (expensive)
    weights payload if the answer is ``"true"``.

    Each stage should use its own command name (e.g.
    ``"pre_send_model_init"``, ``"pre_send_model_learning"``) so
    that the corresponding handler is active in the correct stage.

    Usage::

        gate = ModelGate(ctx.cp, ctx.address, pre_send_command="pre_send_model_learning")
        sent = await gate.send_if_accepted(
            neighbor="node-2:5000",
            weight_command="partial_model",
            contributors=["node-1:5000"],
            round_num=3,
            payload=payload,
        )

    The receiver side remains an ``@on_message`` handler on the stage,
    since acceptance logic is workflow-specific.
    """

    def __init__(self, cp: CommunicationProtocol, address: str, pre_send_command: str) -> None:
        """Initialize the gate with a communication protocol, local address, and pre-send command name."""
        self._cp = cp
        self._address = address
        self._pre_send_command = pre_send_command

    async def send_if_accepted(
        self,
        neighbor: str,
        weight_command: str,
        contributors: list[str],
        round_num: int,
        payload: Any,
    ) -> bool:
        """
        Ask the neighbor whether it wants this model, then send if accepted.

        Args:
            neighbor: Target peer address.
            weight_command: The weight command name (e.g. ``"add_model"``, ``"partial_model"``).
            contributors: List of contributor addresses for dedup checking.
            round_num: Current round number.
            payload: Pre-built weights message (from ``cp.build_weights``).

        Returns:
            True if the model was sent, False if declined or failed.

        """
        try:
            pre_send_msg = self._cp.build_msg(
                self._pre_send_command,
                [weight_command] + contributors,
                round=round_num,
                direct=True,
            )
            response = await self._cp.send(neighbor, pre_send_msg, temporal_connection=True)
            if response != "true":
                logger.debug(self._address, f"⏭️ Skipping model send to {neighbor} - recipient declined")
                return False
            logger.debug(self._address, f"🗣️ Sending model to {neighbor}")
            await self._cp.send(neighbor, payload, temporal_connection=True)
            logger.debug(self._address, f"✅ Sent model to {neighbor}")
            return True
        except Exception as e:
            logger.warning(self._address, f"⚠️ Failed to send model to {neighbor}: {e}")
            return False


def should_accept_model(
    weight_command: str,
    contributors: list[str],
    round: int,
    local_round: int,
    existing_contributors: set[str],
) -> bool:
    """
    Decide whether to accept an incoming model transfer.

    This is the default acceptance logic used by BasicDFL. Other workflows
    should implement their own logic in their ``@on_message("pre_send_model")``
    handler.

    Args:
        weight_command: ``"add_model"`` or ``"partial_model"``.
        contributors: Contributor addresses from the sender.
        round: The sender's round number.
        local_round: This node's current round number.
        existing_contributors: Set of contributor addresses already held locally.

    Returns:
        True if the model should be accepted.

    """
    if weight_command == "add_model":
        return round > local_round
    elif weight_command == "partial_model":
        return bool(set(contributors) - existing_contributors)
    return False
