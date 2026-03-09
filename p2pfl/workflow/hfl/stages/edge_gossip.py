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
"""Edge inter-edge gossip stage for HFL."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from p2pfl.learning.frameworks.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.workflow.engine.message import on_message
from p2pfl.workflow.engine.stage import Stage
from p2pfl.workflow.hfl.context import HFLContext
from p2pfl.workflow.shared.utils import wait_with_timeout

if TYPE_CHECKING:
    from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


class HFLEdgeGossipStage(Stage[HFLContext]):
    """Inter-edge gossip for global model consensus.

    Edge models are buffered with their round number so that messages
    arriving before this stage starts (e.g. while the node is still in
    edge_aggregate_workers) are preserved rather than dropped.
    """

    name = "edge_gossip"

    def __init__(self) -> None:
        """Initialize the gossip stage."""
        super().__init__()
        self._edges_complete = asyncio.Event()
        # Buffer: source -> (round, model).  Persists across stages so that
        # edge models arriving before edge_gossip starts are preserved.
        self._received: dict[str, tuple[int, P2PFLModel]] = {}

    def _all_edge_models_received(self, ctx: HFLContext, current_round: int) -> bool:
        expected = set(ctx.edge_peers) | {ctx.address}
        received = {src for src, (r, _) in self._received.items() if r == current_round}
        return expected.issubset(received)

    # Accept during all edge-lifecycle stages so early arrivals are buffered.
    @on_message(
        "edge_model",
        weights=True,
        during={"edge_local_train", "edge_aggregate_workers", "edge_gossip", "edge_distribute", "round_finished"},
    )
    async def handle_edge_model(
        self,
        source: str,
        round: int,
        weights: bytes,
        contributors: list[str] | None,
        num_samples: int | None,
    ) -> None:
        """Receive an edge model and buffer it."""
        ctx = self.ctx
        current_round = ctx.experiment.round
        if round < current_round:
            logger.debug(ctx.address, f"Dropping stale edge model from {source} (round {round} < {current_round})")
            return
        if round > current_round + 1:
            logger.warning(ctx.address, f"Ignoring future edge model from {source} (round {round} >> {current_round})")
            return
        if contributors is None or num_samples is None:
            raise ValueError("Contributors and num_samples are required")
        try:
            model = ctx.learner.get_model().build_copy(
                params=weights,
                num_samples=num_samples,
                contributors=list(contributors),
            )
            self._received[source] = (round, model)
            logger.debug(ctx.address, f"Edge model buffered from {source} for round {round}.")

            if self._all_edge_models_received(ctx, current_round):
                self._edges_complete.set()
        except DecodingParamsError:
            logger.error(ctx.address, "Error decoding edge model parameters.")
        except ModelNotMatchingError:
            logger.error(ctx.address, "Edge model does not match local model structure.")
        except Exception as e:
            logger.error(ctx.address, f"Error processing edge model: {e}")

    async def run(self) -> str | None:
        """Exchange models with other edges and aggregate for global consensus."""
        ctx = self.ctx
        current_round = ctx.experiment.round
        self._edges_complete.clear()

        # Discard models from previous rounds, keep current-round early arrivals
        self._received = {
            src: (r, m) for src, (r, m) in self._received.items() if r == current_round
        }

        # Store own model (already flattened, contributors=[ctx.address])
        own_model = ctx.learner.get_model()
        self._received[ctx.address] = (current_round, own_model)

        # Send model to all edge peers
        if ctx.edge_peers:
            payload = ctx.cp.build_weights(
                "edge_model",
                current_round,
                own_model.encode_parameters(),
                own_model.get_contributors(),
                own_model.get_num_samples(),
            )
            for edge_addr in ctx.edge_peers:
                await ctx.cp.send(edge_addr, payload)

            # Wait for all edge models
            if not self._all_edge_models_received(ctx, current_round):
                await wait_with_timeout(
                    self._edges_complete,
                    Settings.training.AGGREGATION_TIMEOUT,
                    ctx.address,
                    "Timeout waiting for edge models, proceeding with available.",
                )

            # Aggregate all received models for current round
            models = [m for _, (r, m) in self._received.items() if r == current_round and m is not None]
            if len(models) > 1:
                global_model = ctx.aggregator.aggregate(models)
                ctx.learner.set_model(global_model)

        logger.info(ctx.address, "Edge gossip done.")
        return "edge_distribute"
