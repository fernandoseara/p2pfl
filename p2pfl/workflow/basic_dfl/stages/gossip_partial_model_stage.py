#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
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
"""Gossip partial model stage."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from p2pfl.management.logger import logger
from p2pfl.workflow.engine.stage import Stage

if TYPE_CHECKING:
    from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
    from p2pfl.node import Node
    from p2pfl.workflow.basic_dfl.workflow import BasicPeerState


class GossipPartialModelStage(Stage):
    """GossipPartialModel stage."""

    @staticmethod
    async def execute(peers: dict[str, BasicPeerState], node: Node, candidates: list[str]) -> None:
        """Execute the stage."""
        assert node.workflow is not None
        # Gather all contributors across stored models
        all_contributors: list[str] = []
        for p in peers.values():
            if p.model:
                all_contributors.extend(p.model.get_contributors())
        all_contributors = list(set(all_contributors))

        # Communicate Aggregation
        await node.communication_protocol.broadcast_gossip(
            node.communication_protocol.build_msg(
                "models_aggregated",
                all_contributors,
                round=node.workflow.round,
            )
        )
        await GossipPartialModelStage.__gossip_model_aggregation(peers=peers, node=node, candidates=candidates)

    @staticmethod
    async def __gossip_model_aggregation(
        peers: dict[str, BasicPeerState],
        node: Node,
        candidates: list[str],
    ) -> None:
        """
        Gossip model aggregation.

        CAREFUL:
            - Full connected trainset to increase aggregation speed. On real scenarios, this won't
            be possible, private networks and firewalls.
            - Needed because the trainset can split the networks (and neighbors that are not in the
            trainset won't receive the aggregation).
        """

        def get_model_for_neighbor(n: str) -> tuple[Any, list[str]] | None:
            """Get model payload and contributors for a neighbor, or None if no eligible model."""
            models: list[P2PFLModel] = [p.model for p in peers.values() if p.model is not None]
            peer = peers.get(n)
            aggregation_sources = peer.aggregated_from if peer else []

            # Filter models whose contributors are not in aggregation sources
            eligible_models = [model for model in models if not set(model.get_contributors()).issubset(aggregation_sources)]

            # Select one random eligible model
            model = None
            if eligible_models:
                model = random.choice(eligible_models)

            if model is None:
                logger.info(node.address, f"❔ No models to aggregate for {node.address}.")
                return None

            #
            # NOTE: No estoy seguro de si la forma de acceso a workflow por ejemplo, debiera ser a través de node.workflow o si debería
            # pasarse como argumento o ser algo mas directo quizá.
            #
            # Por ahora lo dejo así, pero es algo a revisar. Lo mismo para communication_protocol, que también se accede a través de node.
            #
            # En cuanto a round, esta en Experiment (con toda la metadata, así que... mismo apunte)
            #

            assert node.workflow is not None
            if node.workflow.experiment is None:
                raise ValueError("Experiment not initialized")

            payload = node.communication_protocol.build_weights(
                "partial_model",
                node.workflow.round,
                model.encode_parameters(),
                model.get_contributors(),
                model.get_num_samples(),
            )
            return (payload, model.get_contributors())

        # Gossip to eligible neighbors
        for neighbor in candidates:
            result = get_model_for_neighbor(neighbor)

            if result is not None:
                payload, contributors = result
                try:
                    # Pre-send confirmation: ask neighbor if they want this model
                    if node.workflow is None:
                        raise ValueError("Workflow not initialized")

                    pre_send_msg = node.communication_protocol.build_msg(
                        "pre_send_model",
                        ["partial_model"] + contributors,
                        round=node.workflow.round,
                        direct=True,
                    )
                    response = await node.communication_protocol.send(neighbor, pre_send_msg, temporal_connection=True)

                    if response != "true":
                        logger.debug(
                            node.address,
                            f"⏭️ Skipping model send to {neighbor} - recipient declined",
                        )
                        continue

                    # Recipient wants the model, send it
                    logger.debug(node.address, f"🗣️ Sending model to {neighbor}")
                    await node.communication_protocol.send(neighbor, payload, temporal_connection=True)
                    logger.debug(node.address, f"✅ Sent model to {neighbor}")
                except Exception as e:
                    logger.warning(node.address, f"⚠️ Failed to send model to {neighbor}: {e}")
