#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/federated_learning_p2p).
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
"""Virtual Node Learner."""

import traceback
from typing import Dict, List, Union

import numpy as np
import ray

from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.learning.frameworks.simulation.actor_pool import VirtualLearnerActor
from p2pfl.learning.frameworks.simulation.placement_group_manager import PlacementGroupManager
from p2pfl.management.logger import logger


class VirtualNodeLearner(Learner):
    """Decorator for the learner to be used in the simulation."""

    def __init__(self, learner: Learner) -> None:
        """Initialize the learner."""
        self.address = "test"
        pg_manager = PlacementGroupManager()
        pg = pg_manager.get_placement_group()

        self.actor = VirtualLearnerActor.options(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
        #).remote(learner_class(**learner_args))
        ).remote(learner)

    async def fit(self) -> None:
        """Fit the model."""
        try:
            await self.actor.fit.remote()
        except Exception as ex:
            logger.error(self.address, traceback.format_exc())
            logger.error(self.address, f"An error occurred during remote fit: {ex}")
            raise ex

    async def train_on_batch(self) -> None:
        """
        Train the model on the next batch manually.

        Returns:
            The model after training on the batch.

        """
        try:
            await self.actor.train_on_batch.remote()
        except Exception as ex:
            logger.error(self.address, traceback.format_exc())
            logger.error(self.address, f"An error occurred during remote train_on_batch: {ex}")
            raise ex

    def interrupt_fit(self) -> None:
        """Interrupt the fit process."""
        # TODO: Need to implement this!
        raise NotImplementedError

    async def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model with actual parameters.

        Returns:
            The evaluation results.

        """
        try:
            return await self.actor.evaluate.remote()
        except Exception as ex:
            logger.error(self.address, traceback.format_exc())
            logger.error(self.address, f"An error occurred during remote evaluation: {ex}")
            raise ex


    # Proxy configuration & lifecycle methods
    def set_P2PFLModel(self, model) -> None: ray.get(self.actor.set_P2PFLModel.remote(model))
    def get_P2PFLModel(self) -> P2PFLModel: return ray.get(self.actor.get_P2PFLModel.remote())
    def set_data(self, data) -> None: ray.get(self.actor.set_data.remote(data))
    def get_data(self): return ray.get(self.actor.get_data.remote())
    def indicate_aggregator(self, aggregator) -> None: ray.get(self.actor.indicate_aggregator.remote(aggregator))
    def get_epochs(self) -> int: return ray.get(self.actor.get_epochs.remote())
    def set_epochs(self, epochs: int) -> None: ray.get(self.actor.set_epochs.remote(epochs))
    def get_steps_per_epoch(self) -> int: return ray.get(self.actor.get_steps_per_epoch.remote())
    def set_steps_per_epoch(self, steps: int) -> None: ray.get(self.actor.set_steps_per_epoch.remote(steps))
    def update_callbacks_with_model_info(self) -> None: ray.get(self.actor.update_callbacks_with_model_info.remote())
    def add_callback_info_to_model(self) -> None: ray.get(self.actor.add_callback_info_to_model.remote())
    def get_framework(self) -> str: return ray.get(self.actor.get_framework.remote())
    def interrupt_fit(self) -> None: raise NotImplementedError
