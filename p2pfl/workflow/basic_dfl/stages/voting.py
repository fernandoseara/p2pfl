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
"""Voting stage for BasicDFL."""

from __future__ import annotations

import asyncio
import math

from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.workflow.basic_dfl.context import BasicDFLContext
from p2pfl.workflow.engine.message import on_message
from p2pfl.workflow.engine.stage import Stage
from p2pfl.workflow.shared.utils import wait_with_timeout


class VotingStage(Stage[BasicDFLContext]):
    """Train-set voting and vote aggregation stage."""

    def __init__(self) -> None:
        """Initialize VotingStage."""
        super().__init__()
        self._votes_complete = asyncio.Event()

    async def run(self) -> str | None:
        """Cast votes, wait for all votes, aggregate, and decide next stage."""
        ctx = self.ctx
        address = ctx.address

        self._votes_complete.clear()

        # Cast votes
        await self._cast_votes(ctx)

        # Wait for all votes with timeout
        await wait_with_timeout(
            self._votes_complete,
            Settings.training.VOTE_TIMEOUT,
            address,
            "Voting timed out, proceeding with available votes.",
        )

        # Aggregate votes
        logger.info(address, "🗳️ Voting finished.")
        self._aggregate_votes(ctx)

        ctx.needs_full_model = ctx.address not in ctx.train_set
        return "learning"

    ###
    #    Voting logic
    ###

    async def _cast_votes(self, ctx: BasicDFLContext) -> None:
        address = ctx.address
        experiment = ctx.experiment

        candidates = list(ctx.peers.keys())
        logger.debug(address, f"{len(candidates)} candidates to train set")
        candidates.sort()

        if experiment.trainset_size is None:
            raise ValueError("trainset_size must be set before voting")
        samples = min(experiment.trainset_size, len(candidates))
        nodes_voted = ctx.generator.sample(candidates, samples)
        weights = [math.floor(ctx.generator.randint(0, 1000) / (i + 1)) for i in range(samples)]

        self_vote = list(zip(nodes_voted, weights, strict=False))
        logger.debug(address, f"🪞🗳️ Self Vote: {self_vote}")
        await self._save_votes(ctx, source=address, round=experiment.round, tmp_votes=self_vote)

        votes_list: list[str | int] = []
        for peer_voted, weight in self_vote:
            votes_list.append(peer_voted)
            votes_list.append(weight)

        await ctx.cp.broadcast_gossip(ctx.cp.build_msg("vote_train_set", votes_list, round=experiment.round))
        logger.info(address, "🗳️ Vote broadcasted.")

    ###
    #    Vote aggregation
    ###

    def _aggregate_votes(self, ctx: BasicDFLContext) -> None:
        address = ctx.address
        experiment = ctx.experiment

        results: dict[str, int] = {}
        for peer in ctx.peers.values():
            for k, v in peer.votes.items():
                results[k] = results.get(k, 0) + v

        ordered = sorted(results.items(), key=lambda x: x[0])
        ordered = sorted(ordered, key=lambda x: x[1], reverse=True)

        if experiment.trainset_size is None:
            raise ValueError("trainset_size must be set before voting")
        top = min(len(ordered), experiment.trainset_size)
        train_set = [i[0] for i in ordered[:top]]

        num_vote_entries = sum(len(p.votes) for p in ctx.peers.values())
        num_peers_voted = sum(1 for p in ctx.peers.values() if p.votes)
        for p in ctx.peers.values():
            p.votes.clear()
        logger.info(address, f"🗳️ Received votes from {num_peers_voted} peers ({num_vote_entries} vote entries).")

        ctx.train_set = [n for n in train_set if n in ctx.peers]
        if not ctx.train_set:
            logger.warning(address, "⚠️ Train set empty after filtering, falling back to self.")
            ctx.train_set = [address]
        logger.info(
            address,
            f"Train set of {len(ctx.train_set)} nodes: {ctx.train_set}",
        )

    ###
    #    Callbacks
    ###

    async def _save_votes(self, ctx: BasicDFLContext, source: str = "", round: int = 0, tmp_votes: list | None = None) -> None:
        if tmp_votes is None:
            return
        if round == ctx.experiment.round:
            peer = ctx.peers.get(source)
            if peer is None:
                logger.warning(ctx.address, f"⚠️ Received votes from unknown peer {source}, ignoring.")
                return
            for train_set_id, vote in tmp_votes:
                peer.votes[train_set_id] = peer.votes.get(train_set_id, 0) + vote
        else:
            logger.debug(ctx.address, f"Rejected stale vote from {source} (msg round={round}, local round={ctx.experiment.round}).")
            return

        if all(p.votes for p in ctx.peers.values()):
            self._votes_complete.set()

    ###
    #    Message handler
    ###

    @on_message("vote_train_set", during={"voting", "round_init", "learning"})
    async def handle_vote_train_set(self, source: str, round: int, *args) -> None:
        """Handle a vote_train_set message by parsing vote pairs and forwarding."""
        if len(args) % 2 != 0:
            raise ValueError("Votes list must contain an even number of elements (peer, weight pairs).")
        votes = [(args[i], int(args[i + 1])) for i in range(0, len(args), 2)]
        await self._save_votes(self.ctx, source, round, votes)
