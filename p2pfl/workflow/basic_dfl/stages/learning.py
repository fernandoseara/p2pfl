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
"""Learning and aggregation stage for BasicDFL."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from p2pfl.learning.frameworks.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.workflow.basic_dfl.context import BasicDFLContext
from p2pfl.workflow.engine.message import on_message
from p2pfl.workflow.engine.stage import Stage
from p2pfl.workflow.shared.evaluate import evaluate_and_broadcast
from p2pfl.workflow.shared.gossiping import ModelGate, should_accept_model
from p2pfl.workflow.shared.utils import wait_with_timeout

if TYPE_CHECKING:
    from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


class LearningStage(Stage[BasicDFLContext]):
    """Evaluate, train, gossip partial models, and aggregate stage."""

    def __init__(self) -> None:
        """Initialize the learning stage."""
        super().__init__()
        self._models_complete = asyncio.Event()

    async def run(self) -> str | None:
        """Evaluate, train, gossip, aggregate, and advance the round."""
        ctx = self.ctx
        self._models_complete.clear()

        address = ctx.address
        learner = ctx.learner
        experiment = ctx.experiment

        # Evaluate
        await evaluate_and_broadcast(ctx)

        # Train
        logger.info(address, "🏋️‍♀️ Training...")
        await learner.fit()
        logger.info(address, "🎓 Training done.")
        await self._save_aggregation(ctx, model=learner.get_model(), source=address)

        # Gossip partial models
        candidates = self._get_partial_gossiping_candidates(ctx)
        if candidates:
            # Gather all contributors
            all_contributors: list[str] = []
            for p in ctx.peers.values():
                if p.model:
                    all_contributors.extend(p.model.get_contributors())
            all_contributors = list(set(all_contributors))

            await ctx.cp.broadcast_gossip(ctx.cp.build_msg("models_aggregated", all_contributors, round=experiment.round))
            await self._gossip_partial_models(ctx, candidates)

        # Wait for all models with timeout
        if not self._all_models_received(ctx):
            await wait_with_timeout(
                self._models_complete,
                Settings.training.AGGREGATION_TIMEOUT,
                address,
                "Aggregation timed out, proceeding with available models.",
            )

        # Aggregate
        aggregator = ctx.aggregator
        agg_model = aggregator.aggregate([p.model for p in ctx.peers.values() if p.model is not None])
        learner.set_model(agg_model)
        experiment.increase_round(address)

        logger.info(address, "Aggregation finished.")
        logger.info(address, f"Round {experiment.round} finished.")

        ctx.needs_full_model = False
        return "round_init"

    # -- Gossiping --

    async def _gossip_partial_models(self, ctx: BasicDFLContext, candidates: list[str]) -> None:
        address = ctx.address
        experiment = ctx.experiment

        for neighbor in candidates:
            models = [p.model for p in ctx.peers.values() if p.model is not None]
            peer = ctx.peers.get(neighbor)
            aggregation_sources = peer.aggregated_from if peer else []
            eligible = [m for m in models if not set(m.get_contributors()).issubset(aggregation_sources)]

            if not eligible:
                logger.info(address, f"No models to aggregate for {address}.")
                continue

            model = ctx.generator.choice(eligible)
            payload = ctx.cp.build_weights(
                "partial_model",
                experiment.round,
                model.encode_parameters(),
                model.get_contributors(),
                model.get_num_samples(),
            )
            gate = ModelGate(ctx.cp, address, pre_send_command="pre_send_model_learning")
            await gate.send_if_accepted(
                neighbor=neighbor,
                weight_command="partial_model",
                contributors=model.get_contributors(),
                round_num=experiment.round,
                payload=payload,
            )

    # -- Condition helpers --

    def _get_partial_gossiping_candidates(self, ctx: BasicDFLContext) -> list[str]:
        address = ctx.address
        train_set = set(ctx.train_set)
        other_nodes = train_set - {address}
        # A neighbor is a candidate if it hasn't yet aggregated all contributors from the train set
        candidates = [n for n in other_nodes if not train_set.issubset(set(p.aggregated_from) if (p := ctx.peers.get(n)) else set())]
        logger.debug(address, f"Candidates to gossip to: {candidates}")
        return candidates

    def _all_models_received(self, ctx: BasicDFLContext) -> bool:
        return len(ctx.train_set) == sum(1 for p in ctx.peers.values() if p.model is not None)

    # -- State update callbacks --

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
                logger.warning(ctx.address, f"Ignoring aggregated_models from unknown peer {source}")
                return
            peer.aggregated_from.extend(aggregated_models)
            logger.debug(
                ctx.address,
                f"Aggregated models received from {source}: {aggregated_models}",
            )
        else:
            logger.error(
                ctx.address,
                f"Aggregated models not received from {source}: {aggregated_models} (expected {ctx.experiment.round})",
            )

    async def _save_aggregation(self, ctx: BasicDFLContext, model: P2PFLModel | None = None, source: str = "") -> None:
        if model is None:
            return
        peer = ctx.peers.get(source)
        if peer is None:
            logger.warning(ctx.address, f"Ignoring model from unknown peer {source}")
            return
        peer.model = model
        logger.debug(ctx.address, f"Model received from {source}: {model}")

        if self._all_models_received(ctx):
            self._models_complete.set()

    # -- Message handlers --

    @on_message("models_aggregated", during={"learning", "voting"})
    async def handle_models_aggregated(self, source: str, round: int, *args) -> None:
        """Handle a models_aggregated message by forwarding contributors."""
        await self._save_aggregated_models(self.ctx, source, round, list(args))

    @on_message("pre_send_model_learning", during={"learning", "voting"})
    async def handle_pre_send_model_learning(self, source: str, round: int, *args) -> str:
        """Handle a pre_send_model_learning request by checking if the model should be accepted."""
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

    @on_message("partial_model", weights=True)
    async def handle_partial_model(
        self,
        source: str,
        round: int,
        weights: bytes,
        contributors: list[str] | None,
        num_samples: int | None,
    ) -> None:
        """Handle a partial_model message by decoding and aggregating the received model."""
        ctx = self.ctx
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
            logger.error(ctx.address, "Error decoding parameters.")
        except ModelNotMatchingError:
            logger.error(ctx.address, "Models not matching.")
        except Exception as e:
            logger.error(ctx.address, f"Unknown error adding model: {e}")
