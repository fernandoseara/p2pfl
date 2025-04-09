#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2022 Pedro Guijas Bravo.
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
"""Train stage."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Set, Type, Union

from p2pfl.communication.commands.message.models_agregated_command import ModelsAggregatedCommand
from p2pfl.communication.commands.weights.partial_model_command import PartialModelCommand
from p2pfl.learning.aggregators.aggregator import NoModelsToAggregateError
from p2pfl.management.logger import logger
from p2pfl.stages.stage import EarlyStopException, Stage, check_early_stop

if TYPE_CHECKING:
    from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
    from p2pfl.learning.aggregators.aggregator import Aggregator
    from p2pfl.learning.frameworks.learner import Learner
    from p2pfl.node_state import NodeState


class TrainStage(Stage):
    """Train stage."""

    @staticmethod
    async def execute(
        state: NodeState,
        communication_protocol: CommunicationProtocol,
        learner: Learner,
        aggregator: Aggregator
        ) -> None:
        """Execute the stage."""
        try:
            # Train
            logger.info(state.addr, "🏋️‍♀️ Training...")
            await learner.fit()
            logger.info(state.addr, "🎓 Training done.")

            # Aggregate Model
            models_added = aggregator.add_model(learner.get_model())

            # send model added msg ---->> redundant (a node always owns its model)
            # TODO: print("Broadcast redundante")
            await communication_protocol.broadcast(
                communication_protocol.build_msg(
                    ModelsAggregatedCommand.get_name(),
                    models_added,
                    round=state.round,
                )
            )
            await TrainStage.__gossip_model_aggregation(state, communication_protocol, aggregator)

            # Set aggregated model
            agg_model = aggregator.wait_and_get_aggregation()
            learner.set_model(agg_model)
        except EarlyStopException:
            return None

    @staticmethod
    async def __gossip_model_aggregation(
        state: NodeState,
        communication_protocol: CommunicationProtocol,
        aggregator: Aggregator,
    ) -> None:
        """
        Gossip model aggregation.

        CAREFULL:
            - Full connected trainset to increase aggregation speed. On real scenarios, this won't
            be possible, private networks and firewalls.
            - Needed because the trainset can split the networks (and neighbors that are not in the
            trainset won't receive the aggregation).
        """

        # Anonymous functions
        def early_stopping_fn():
            return state.round is None

        def get_candidates_fn() -> list[str]:
            candidates = set(state.train_set) - {state.addr}
            return [n for n in candidates if len(TrainStage.__get_remaining_nodes(n, state)) != 0]

        def status_fn() -> Any:
            return [
                (
                    n,
                    TrainStage.__get_aggregated_models(n, state),
                )  # reemplazar por Aggregator - borrarlo de node
                for n in communication_protocol.get_neighbors(only_direct=False)
                if (n in state.train_set)
            ]

        def model_fn(node: str) -> Any:
            try:
                model = aggregator.get_model(TrainStage.__get_aggregated_models(node, state))
            except NoModelsToAggregateError:
                logger.info(state.addr, f"❔ No models to aggregate for {node}.")
                return None
            if state.round is None:
                raise Exception("Round not initialized.")
            return communication_protocol.build_weights(
                PartialModelCommand.get_name(),
                state.round,
                model.encode_parameters(),
                model.get_contributors(),
                model.get_num_samples(),
            )

        # Gossip
        await communication_protocol.gossip_weights(
            early_stopping_fn,
            get_candidates_fn,
            status_fn,
            model_fn,
            create_connection=True,
        )

    @staticmethod
    def __get_aggregated_models(node: str, state: NodeState) -> List[str]:
        try:
            return state.models_aggregated[node]
        except KeyError:
            return []

    @staticmethod
    def __get_remaining_nodes(node: str, state: NodeState) -> Set[str]:
        return set(state.train_set) - set(TrainStage.__get_aggregated_models(node, state))
