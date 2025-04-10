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

"""Stage factory."""

import asyncio

from statemachine import Event, State

from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.stages.base_node.aggregation_finished_stage import AggregationFinishedStage
from p2pfl.stages.base_node.evaluate_stage import EvaluateStage
from p2pfl.stages.base_node.gossip_final_model_stage import GossipFinalModelStage
from p2pfl.stages.base_node.initialize_model_stage import InitializeModelStage
from p2pfl.stages.base_node.round_finished_stage import RoundFinishedStage
from p2pfl.stages.base_node.start_learning_stage import StartLearningStage
from p2pfl.stages.base_node.train_stage import TrainStage
from p2pfl.stages.base_node.training_finished_stage import TrainingFinishedStage
from p2pfl.stages.base_node.vote_train_set_stage import VoteTrainSetStage
from p2pfl.stages.base_node.wait_start_learning import WaitingForStartStage
from p2pfl.stages.workflows import TrainingWorkflow


class BasicDFLWorkflow(TrainingWorkflow):
    """
    Class to run a federated learning workflow.

    It runs a state machine with the following states:
    - waiting_for_training_start: Wait for the training to start.
    - starting_training: Start the training.
    - voting: Vote for the train set.
    - evaluating: Evaluate the model.
    - training: Train the model.
    - aggregating: Aggregate the model.
    - waiting_for_aggregation: Wait for the aggregation to finish or timeout.
    - aggregation_finished: Finish the aggregation.
    - gossiping: Gossip the model.
    - round_finished: Finish the round.
    - training_finished: Finish the training.
    The workflow is implemented as a state machine using the
    statemachine library. The states are defined as class attributes,
    and the transitions between states are defined as class methods.

    The model of the workflow is corresponding to the Node class.
    """

    # States
    waiting_for_training_start = State("Start", initial=True)
    starting_training = State()
    initializing_model = State()
    voting = State()
    evaluating = State()
    training = State()
    aggregating = State()
    waiting_for_aggregation = State()
    aggregation_finished = State()
    gossiping = State()
    round_finished = State()
    training_finished = State("End", final=True)

    # Events and Transitions
    start_training = waiting_for_training_start.to(starting_training)
    initialize_model = starting_training.to(initializing_model)
    vote = initializing_model.to(voting)
    train = voting.to(evaluating, cond="in_train_set") \
        | voting.to(waiting_for_aggregation, cond="!in_train_set") \
        | evaluating.to(training)
    aggregate = training.to(aggregating)
    finish_aggregation = aggregating.to(aggregation_finished) | waiting_for_aggregation.to(aggregation_finished)
    gossip = aggregation_finished.to(gossiping)
    finish_round = gossiping.to(round_finished)
    step = round_finished.to(voting, cond="!is_total_rounds_reached") \
        | round_finished.to(training_finished, cond="is_total_rounds_reached")

    def __init__(self, *args, **kwargs):
        """Initialize the workflow."""
        self.state_changed = asyncio.Event()
        super().__init__(*args, **kwargs)

    ###################
    # STATE CALLBACKS #
    ###################
    @TrainingWorkflow.run_in_executor
    async def on_enter_waiting_for_training_start(self):
        """Wait for the training to start."""
        await WaitingForStartStage.execute()

    @TrainingWorkflow.run_in_executor
    async def on_enter_starting_training(self,
            experiment_name: str,
            rounds: int,
            epochs: int,
            trainset_size: int,
        ):
        """Start the training."""
        self.is_running = True
        await StartLearningStage.execute(
            experiment_name=experiment_name,
            rounds=rounds,
            epochs=epochs,
            trainset_size=trainset_size,
            communication_protocol=self.node.communication_protocol,
            state=self.node.state,
            learner=self.node.learner,
        )

    @TrainingWorkflow.run_in_executor
    async def on_enter_initializing_model(self, round: int, weights: bytes):
        """Initialize the model."""
        await InitializeModelStage.execute(
            round=round,
            weights=weights,
            state=self.node.state,
            learner=self.node.learner,
        )
        await self.vote()

    @TrainingWorkflow.run_in_executor
    async def on_enter_voting(self):
        """Vote for the train set."""
        await VoteTrainSetStage.execute(
            state=self.node.state,
            communication_protocol=self.node.communication_protocol,
            aggregator=self.node.aggregator,
            generator=self.node.generator
        )
        await self.train()

    @TrainingWorkflow.run_in_executor
    async def on_enter_evaluating(self):
        """Evaluate the model."""
        await EvaluateStage.execute(
            state=self.node.state,
            communication_protocol=self.node.communication_protocol,
            learner=self.node.learner)
        await self.train()

    @TrainingWorkflow.run_in_executor
    async def on_enter_training(self):
        """Train the model."""
        await TrainStage.execute(
            state=self.node.state,
            communication_protocol=self.node.communication_protocol,
            learner=self.node.communication_protocol,
            aggregator=self.node.aggregator
        )
        await self.aggregate()

    @TrainingWorkflow.run_in_executor
    async def on_enter_waiting_for_aggregation(self):
        """Wait for the aggregation to finish or timeout."""
        try:
            await asyncio.wait_for(self.state_changed.wait(), timeout=Settings.training.AGGREGATION_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning(self.node.state.addr, "⏰ Aggregation timeout occurred.")
            await self.gossip()

    @TrainingWorkflow.run_in_executor
    async def on_enter_aggregation_finished(self):
        """Finish the aggregation."""
        await AggregationFinishedStage.execute(
            state=self.node.state,
            communication_protocol=self.node.communication_protocol
        )
        await self.gossip()

    @TrainingWorkflow.run_in_executor
    async def on_enter_gossiping(self):
        """Gossip the model."""
        await GossipFinalModelStage.execute(
            state=self.node.state,
            communication_protocol=self.node.communication_protocol,
            learner=self.node.learner
        )
        await self.finish_round()

    @TrainingWorkflow.run_in_executor
    async def on_enter_round_finished(self):
        """Finish the round."""
        await RoundFinishedStage.execute(
            state=self.node.state,
            aggregator=self.node.aggregator
        )

        await self.step()

    @TrainingWorkflow.run_in_executor
    async def on_enter_training_finished(self):
        """Finish the training."""
        await TrainingFinishedStage.execute(
            state=self.node.state,
            communication_protocol=self.node.communication_protocol,
            learner=self.node.learner
        )
        self.is_running = False

    ##############
    # CONDITIONS #
    ##############
    def in_train_set(self):
        """Check if the node is in the train set."""
        return self.node.state.addr in self.node.state.train_set

    def is_total_rounds_reached(self):
        """Check if the total rounds have been reached."""
        return self.node.state.round >= self.node.state.total_rounds

    #####################
    # GENERAL CALLBACKS #
    #####################
    def on_transition(self, event_data, event: Event):
        """Handle the transition event."""
        super().on_transition(event_data, event)
        self.state_changed.set()
        self.state_changed.clear()


