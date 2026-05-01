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
"""Loop-based gossip stage for BasicDFL with partial aggregation support."""

from __future__ import annotations

import asyncio
import contextlib
import random
from typing import TYPE_CHECKING

from p2pfl.learning.frameworks.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.workflow.basic_dfl.context import BasicDFLContext
from p2pfl.workflow.engine.message import on_message
from p2pfl.workflow.engine.stage import Stage
from p2pfl.workflow.shared.gossiping import ModelGate, should_accept_model

if TYPE_CHECKING:
    from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


class LearningGossipLoopStage(Stage[BasicDFLContext]):
    """
    Loop-based gossip that periodically sends partial aggregations to peers.

    Runs a periodic gossip loop that:
    - Re-checks candidates each iteration (picks up newly received models)
    - Sends partial aggregations (multiple models combined) when the
      aggregator supports it
    - Uses MODELS_PER_ROUND, MODELS_PERIOD, and EXIT_ON_X_EQUAL_ROUNDS settings
    - Retries until all models are collected, convergence is detected, or no candidates remain
    """

    def __init__(self) -> None:
        """Initialize gossip loop stage with pending transfer tracking."""
        # Track in-flight transfers to prevent concurrent redundant sends.
        # When a pre-send is accepted, we mark the offered contributors as pending
        # so that concurrent pre-sends from other senders for the same contributors
        # are rejected. Cleared when the actual model arrives or on timeout.
        self._pending_contributors: dict[str, tuple[set[str], float]] = {}

    async def run(self) -> str | None:
        """Gossip partial models in a loop, wait for all models, then proceed to aggregate."""
        ctx = self.ctx

        # Clear pending state from previous round
        self._pending_contributors.clear()

        # Check if all contributors are already covered (e.g., single trainer)
        if self._has_all_contributors(ctx):
            ctx.models_complete.set()

        await self._gossip_loop(ctx)
        if not ctx.models_complete.is_set():
            logger.info(ctx.address, "⚠️ Gossip loop finished without all contributors — proceeding with partial aggregation.")

        return "learning_aggregate"

    async def _gossip_loop(self, ctx: BasicDFLContext) -> None:
        """Loop-based gossip with retries, partial aggregation, and dedup."""
        sent_contributors: dict[str, frozenset[str]] = {}
        prev_coverage: frozenset[str] = frozenset()
        unchanged_iterations = 0
        gate = ModelGate(ctx.cp, ctx.address, pre_send_command="pre_send_model_learning")

        # Jitter to stagger starts — enables partial aggregation by creating timing asymmetry
        await asyncio.sleep(random.uniform(0, Settings.gossip.MODELS_PERIOD))

        while True:
            start_time = asyncio.get_event_loop().time()

            candidates = self._get_candidates(ctx)

            if not candidates and ctx.models_complete.is_set():
                logger.info(ctx.address, "🤫 Gossip finished — all models collected, no more candidates.")
                break
            if not candidates:
                break

            # Safety exit: if coverage hasn't changed for X iterations, a trainer
            # is likely stuck/dead — stop waiting and aggregate what we have.
            current_coverage = frozenset(self._get_all_contributors(ctx))
            if current_coverage == prev_coverage:
                unchanged_iterations += 1
                if unchanged_iterations >= Settings.gossip.EXIT_ON_X_EQUAL_ROUNDS:
                    logger.warning(
                        ctx.address,
                        f"⚠️ Gossip stalled — no new models for {unchanged_iterations} iterations, proceeding with partial aggregation.",
                    )
                    break
            else:
                unchanged_iterations = 0
                prev_coverage = current_coverage

            # Sample and send
            sample_size = min(Settings.gossip.MODELS_PER_ROUND, len(candidates))
            sample = ctx.generator.sample(candidates, sample_size)

            for neighbor in sample:
                await self._send_model_to(ctx, gate, neighbor, sent_contributors)

            # Wait for the remainder of the period (same pattern as Gossiper._run)
            elapsed = asyncio.get_event_loop().time() - start_time
            await asyncio.sleep(max(0, Settings.gossip.MODELS_PERIOD - elapsed))

    ###
    #    Gossip helpers
    ###

    async def _send_model_to(
        self,
        ctx: BasicDFLContext,
        gate: ModelGate,
        neighbor: str,
        sent_contributors: dict[str, frozenset[str]],
    ) -> bool:
        """
        Build and send a (possibly partially aggregated) model to a neighbor.

        The expensive work (aggregation + parameter encoding) is deferred until
        after the lightweight pre-send gate confirms the neighbor wants the model.

        Returns True if a new model was sent, False if nothing new to offer.
        """
        peer = ctx.peers.get(neighbor)
        aggregated_by_neighbor = peer.aggregated_from if peer else []

        # Collect models whose contributors are NOT already known by neighbor
        models = [p.model for p in ctx.peers.values() if p.model is not None]
        eligible = [m for m in models if not set(m.get_contributors()).issubset(aggregated_by_neighbor)]

        if not eligible:
            return False

        # Determine contributors we'd offer (cheap) without aggregating yet
        use_partial = ctx.aggregator.partial_aggregation and len(eligible) > 1
        if use_partial:
            offer_contributors = list({c for m in eligible for c in m.get_contributors()})
        else:
            ctx.generator.shuffle(eligible)
            offer_contributors = eligible[0].get_contributors()

        # Dedup: skip if no truly new contributors for this neighbor.
        # Check against both what we already sent AND what the neighbor reports having.
        contrib_set = frozenset(offer_contributors)
        previously_sent = sent_contributors.get(neighbor, frozenset())
        known_by_neighbor = set(aggregated_by_neighbor) | previously_sent
        new_for_neighbor = contrib_set - known_by_neighbor
        if not new_for_neighbor:
            return False

        # Lightweight pre-send gate — ask neighbor before doing expensive work
        accepted = await gate.check_acceptance(neighbor, "partial_model", offer_contributors, ctx.experiment.round)
        if not accepted:
            return False

        # Accepted — now aggregate, encode, and send
        model = ctx.aggregator.aggregate(eligible) if use_partial else eligible[0]

        payload = ctx.cp.build_weights(
            "partial_model",
            ctx.experiment.round,
            model.encode_parameters(),
            model.get_contributors(),
            model.get_num_samples(),
        )
        try:
            await ctx.cp.send(neighbor, payload, temporal_connection=True)
            sent_contributors[neighbor] = previously_sent | contrib_set
            return True
        except Exception as e:
            logger.warning(ctx.address, f"⚠️ Failed to send model to {neighbor}: {e}")
            return False

    ###
    #    Condition helpers
    ###

    def _get_candidates(self, ctx: BasicDFLContext) -> list[str]:
        """Return trainers that still need models from us."""
        train_set = set(ctx.train_set)
        other_nodes = train_set - {ctx.address}
        local_round = ctx.experiment.round
        candidates = []
        for n in other_nodes:
            peer = ctx.peers.get(n)
            if peer is None:
                continue
            # Skip peers that already advanced to a higher round
            if peer.round_number > local_round:
                continue
            if not train_set.issubset(set(peer.aggregated_from)):
                candidates.append(n)
        logger.debug(ctx.address, f"Gossip candidates: {candidates}")
        return candidates

    def _get_all_contributors(self, ctx: BasicDFLContext) -> set[str]:
        """Return the union of all contributors across received models."""
        contributors: set[str] = set()
        for p in ctx.peers.values():
            if p.model is not None:
                contributors.update(p.model.get_contributors())
        return contributors

    def _has_all_contributors(self, ctx: BasicDFLContext) -> bool:
        """Check if we have contributions from all trainers in the train set."""
        return set(ctx.train_set).issubset(self._get_all_contributors(ctx))

    async def _broadcast_coverage(
        self, ctx: BasicDFLContext, coverage: set[str] | frozenset[str], targets: list[str] | None = None
    ) -> None:
        """Broadcast current contributor coverage to peers so they stop sending redundant models."""
        targets = targets if targets is not None else [n for n in ctx.train_set if n != ctx.address]
        msg = ctx.cp.build_msg("models_aggregated", list(coverage), round=ctx.experiment.round)
        for n in targets:
            with contextlib.suppress(Exception):
                await ctx.cp.send(n, msg, temporal_connection=True)

    ###
    #    State update callbacks
    ###

    async def _save_aggregated_models(
        self,
        ctx: BasicDFLContext,
        source: str = "",
        round: int = 0,
        aggregated_models: list[str] | None = None,
    ) -> None:
        if aggregated_models is None:
            return
        if round == ctx.experiment.round:
            peer = ctx.peers.get(source)
            if peer is None:
                logger.warning(ctx.address, f"⚠️ Ignoring aggregated_models from unknown peer {source}")
                return
            peer.aggregated_from.extend(aggregated_models)
            logger.debug(
                ctx.address,
                f"Aggregated models received from {source}: {aggregated_models}",
            )
        else:
            logger.debug(
                ctx.address,
                f"Ignoring stale models_aggregated from {source} (round {round}, local {ctx.experiment.round})",
            )

    async def _save_aggregation(self, ctx: BasicDFLContext, model: P2PFLModel | None = None, source: str = "") -> None:
        if model is None:
            return
        peer = ctx.peers.get(source)
        if peer is None:
            logger.warning(ctx.address, f"⚠️ Ignoring model from unknown peer {source}")
            return

        # Redundant models are already rejected at the pre-send gate
        # (should_accept_model), so here we only guard against replacements
        # that would reduce global coverage.
        old_model = peer.model
        if old_model is not None:
            old_coverage = self._get_all_contributors(ctx)
            peer.model = model
            new_coverage = self._get_all_contributors(ctx)
            if len(new_coverage) <= len(old_coverage):
                peer.model = old_model
                logger.debug(
                    ctx.address,
                    f"Rejected model from {source} — no new contributors ({len(old_coverage)} → {len(new_coverage)})",
                )
                return
        else:
            peer.model = model

        # Clear models from other peers fully subsumed by the new model
        # to prevent double-counting in final aggregation
        new_contribs = set(model.get_contributors())
        for addr, p in ctx.peers.items():
            if addr == source or p.model is None:
                continue
            if set(p.model.get_contributors()).issubset(new_contribs):
                logger.debug(ctx.address, f"Clearing subsumed model from peer {addr} (covered by {source})")
                p.model = None

        new_coverage = self._get_all_contributors(ctx)
        total = len(ctx.train_set)
        contributors = model.get_contributors()
        logger.info(ctx.address, f"🧩 Model received ({len(new_coverage)}/{total}) - from {source} - {len(contributors)} contributors.")
        logger.debug(ctx.address, f"Contributors: {contributors}")

        if self._has_all_contributors(ctx):
            ctx.models_complete.set()

        await self._broadcast_coverage(ctx, new_coverage)

    ###
    #    Message handlers
    ###

    @on_message("models_aggregated", during={"learning_.*", "voting", "round_init"})
    async def handle_models_aggregated(self, source: str, round: int, *args) -> None:
        """Handle a models_aggregated message by forwarding contributors."""
        await self._save_aggregated_models(self.ctx, source, round, list(args))

    @on_message("pre_send_model_learning", during={"learning_.*", "voting", "round_init"})
    async def handle_pre_send_model_learning(self, source: str, round: int, *args) -> str:
        """Handle a pre_send_model_learning request by checking if the model should be accepted."""
        if not args:
            return "false"
        weight_command = args[0]
        contributors = list(args[1:]) if len(args) > 1 else []

        # Clean up expired pending entries
        now = asyncio.get_event_loop().time()
        expired = [s for s, (_, ts) in self._pending_contributors.items() if now - ts > Settings.gossip.MODEL_GATE_TIMEOUT]
        for s in expired:
            del self._pending_contributors[s]

        existing: set[str] = set()
        coverage_without_source: set[str] = set()
        for addr, p in self.ctx.peers.items():
            if p.model:
                contribs = set(p.model.get_contributors())
                existing.update(contribs)
                if addr != source:
                    coverage_without_source.update(contribs)

        # Include pending (in-flight) contributors from other accepted pre-sends.
        # This prevents concurrent senders from both getting accepted for overlapping contributors.
        for s, (pending_contribs, _) in self._pending_contributors.items():
            existing.update(pending_contribs)
            if s != source:
                coverage_without_source.update(pending_contribs)

        accepted = should_accept_model(
            weight_command=weight_command,
            contributors=contributors,
            round=round,
            local_round=self.ctx.experiment.round,
            existing_contributors=existing,
            coverage_without_source=coverage_without_source,
        )

        if accepted:
            # Mark these contributors as pending (in-flight)
            self._pending_contributors[source] = (set(contributors), now)

        return "true" if accepted else "false"

    @on_message("partial_model", weights=True, during={"learning_.*", "voting", "round_init"})
    async def handle_partial_model(
        self,
        source: str,
        round: int,
        weights: bytes,
        contributors: list[str] | None,
        num_samples: int | None,
    ) -> None:
        """Handle a partial_model message by decoding and aggregating the received model."""
        # Clear pending state — the actual model arrived (or we're about to process it)
        self._pending_contributors.pop(source, None)

        ctx = self.ctx
        if round != ctx.experiment.round:
            logger.warning(ctx.address, f"⚠️ Ignoring partial_model from {source} (round {round}, local {ctx.experiment.round})")
            return
        if contributors is None or num_samples is None:
            raise ValueError("Contributors and num_samples are required")
        try:
            model = ctx.learner.get_model().build_copy(
                params=weights,
                num_samples=num_samples,
                contributors=list(contributors),
            )
            await self._save_aggregation(ctx, model, source)
        except DecodingParamsError:
            logger.error(ctx.address, "❌ Error decoding parameters.")
        except ModelNotMatchingError:
            logger.error(ctx.address, "❌ Models not matching.")
        except Exception as e:
            logger.error(ctx.address, f"❌ Unknown error adding model: {e}")
