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
"""Round initialization stage for BasicDFL."""

from __future__ import annotations

import asyncio

from p2pfl.learning.frameworks.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.workflow.basic_dfl.context import BasicDFLContext
from p2pfl.workflow.engine.message import on_message
from p2pfl.workflow.engine.stage import Stage
from p2pfl.workflow.shared.gossiping import ModelGate, should_accept_model
from p2pfl.workflow.shared.utils import wait_with_timeout


class RoundInitStage(Stage[BasicDFLContext]):
    """Round initialization, full-model gossiping, and round readiness stage."""

    def __init__(self) -> None:
        """Initialize round-init stage events."""
        self._full_model_ready = asyncio.Event()
        self._all_rounds_synced = asyncio.Event()

    async def run(self) -> str | None:
        """Initialize round, gossip full model, wait for peers."""
        ctx = self.ctx
        self._full_model_ready.clear()
        self._all_rounds_synced.clear()

        experiment = ctx.experiment
        address = ctx.address

        # Initiator path vs non-initiator path
        if experiment.is_initiator and experiment.round == 0:
            logger.info(address, "Initializing the network as initiator.")
            experiment.increase_round(address)
        elif ctx.needs_full_model:
            if not self._has_received_full_model(ctx):
                await wait_with_timeout(
                    self._full_model_ready,
                    Settings.training.SYNCHRONIZATION_TIMEOUT,
                    address,
                    "Timeout waiting for full model. Proceeding anyway.",
                )

        # Update round
        for p in ctx.peers.values():
            p.reset_round()
        logger.info(address, f"Round {experiment.round} of {experiment.total_rounds} started.")
        await self._save_peer_round_updated(ctx, source=address, round=experiment.round)
        await ctx.cp.broadcast_gossip(ctx.cp.build_msg("peer_round_updated", round=experiment.round))

        # Gossip full model to peers with lower round number
        candidates = self._get_full_gossiping_candidates(ctx)
        if candidates:
            logger.info(address, "🗣️ Gossiping full model.")
            encoded_model = ctx.learner.get_model().encode_parameters()
            round_num = experiment.round
            payload = ctx.cp.build_weights("add_model", round_num, encoded_model)

            gate = ModelGate(ctx.cp, address, pre_send_command="pre_send_model_init")
            for neighbor in candidates:
                await gate.send_if_accepted(
                    neighbor=neighbor,
                    weight_command="add_model",
                    contributors=[address],
                    round_num=round_num,
                    payload=payload,
                )

        # Wait for all peers to be at current round
        if not self._all_models_initialized(ctx):
            await wait_with_timeout(
                self._all_rounds_synced,
                Settings.training.SYNCHRONIZATION_TIMEOUT,
                address,
                "Timeout waiting for all peers to sync rounds. Proceeding anyway.",
            )

        logger.debug(address, "Round initialized.")

        if self._total_rounds_reached(ctx):
            return "finish"
        return "voting"

    # -- Condition helpers --

    def _get_full_gossiping_candidates(self, ctx: BasicDFLContext) -> list[str]:
        fixed_round = ctx.experiment.round
        candidates = [
            n for n in ctx.cp.get_neighbors(only_direct=True) if (peer := ctx.peers.get(n)) is not None and peer.round_number < fixed_round
        ]
        logger.debug(ctx.address, f"Candidates to gossip to: {candidates}")
        return candidates

    def _all_models_initialized(self, ctx: BasicDFLContext) -> bool:
        return all(p.round_number == ctx.experiment.round for p in ctx.peers.values())

    def _total_rounds_reached(self, ctx: BasicDFLContext) -> bool:
        return ctx.experiment.is_complete()

    def _has_received_full_model(self, ctx: BasicDFLContext) -> bool:
        peer = ctx.peers.get(ctx.address)
        if peer is None:
            return False
        return peer.round_number >= ctx.experiment.round

    # -- State update callbacks --

    async def _save_peer_round_updated(self, ctx: BasicDFLContext, source: str = "", round: int = 0) -> None:
        local_round = ctx.experiment.round
        if round in [local_round, local_round + 1]:
            peer = ctx.peers.get(source)
            if peer is None:
                logger.warning(ctx.address, f"Received round update from unknown peer {source}, ignoring.")
                return
            peer.round_number = round
            logger.debug(ctx.address, f"Peer round updated: {source} -> {round}")
        else:
            logger.error(
                ctx.address,
                f"Peer round not updated: {source} -> {round} (local round: {local_round})",
            )
        if self._all_models_initialized(ctx):
            self._all_rounds_synced.set()

    async def _save_full_model(self, ctx: BasicDFLContext, source: str = "", round: int = 0, weights: bytes = b"") -> None:
        logger.info(ctx.address, "Full model received.")
        try:
            local_round = ctx.experiment.round
            if round != local_round + 1:
                logger.warning(
                    ctx.address,
                    f"Full model round {round} does not match local round {local_round}. Ignoring.",
                )
                return
            ctx.learner.set_model(weights)
            ctx.experiment.increase_round(ctx.address)
            logger.info(ctx.address, "Model Weights Initialized")
            self._full_model_ready.set()
        except DecodingParamsError:
            logger.error(ctx.address, "Error decoding parameters.")
            return
        except ModelNotMatchingError:
            logger.error(ctx.address, "Models not matching.")
            return
        except Exception as e:
            logger.error(ctx.address, f"Unknown error adding model: {e}")
            return

    # -- Message handlers --

    @on_message("peer_round_updated", during={"round_init", "learning", "voting"})
    async def handle_peer_round_updated(self, source: str, round: int, *args) -> None:
        """Handle a peer_round_updated message."""
        await self._save_peer_round_updated(self.ctx, source, round)

    @on_message("pre_send_model_init", during={"round_init"})
    async def handle_pre_send_model_init(self, source: str, round: int, *args) -> str:
        """Handle a pre_send_model_init request for full model gossiping."""
        if not args:
            return "false"
        weight_command = args[0]
        contributors = list(args[1:]) if len(args) > 1 else []

        existing: set[str] = set()
        for p in self.ctx.peers.values():
            if p.model:
                existing.update(p.model.get_contributors())

        accepted = should_accept_model(
            weight_command=weight_command,
            contributors=contributors,
            round=round,
            local_round=self.ctx.experiment.round,
            existing_contributors=existing,
        )
        return "true" if accepted else "false"

    @on_message("add_model", weights=True, during={"round_init", "learning"})
    async def handle_add_model(
        self,
        source: str,
        round: int,
        weights: bytes,
        contributors: list[str] | None,
        num_samples: int | None,
    ) -> None:
        """Handle an add_model message containing a full model from a peer."""
        await self._save_full_model(self.ctx, source, round, weights)
