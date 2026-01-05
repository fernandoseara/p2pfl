#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2024 Pedro Guijas Bravo.
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

"""XGBoost Learner for P2PFL."""

import numpy as np
import xgboost as xgb
from sklearn.exceptions import NotFittedError
from sklearn.metrics import f1_score

from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks import Framework
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.learning.frameworks.xgboost.xgboost_dataset import XGBoostExportStrategy
from p2pfl.utils.node_component import allow_no_addr_check


class XGBoostLearner(Learner):
    """
    Learner implementation using XGBoost boosting framework.

    Provides training and evaluation functionality for XGBoost models
    in the P2PFL federated learning system.

    Args:
        model (P2PFLModel | None): Wrapped XGBoostModel instance.
        data (P2PFLDataset | None): P2PFLDataset for train/eval split.
        aggregator (Aggregator | None): Aggregator to use for model updates.

    """

    def __init__(self, model: P2PFLModel | None = None, data: P2PFLDataset | None = None, aggregator: Aggregator | None = None) -> None:
        """
        Initialize the XGBoost learner.

        Args:
            model (P2PFLModel | None): The model to train (default is None).
            data (P2PFLDataset | None): The dataset for training and evaluation
                (default is None).
            aggregator (Aggregator | None): The aggregator for model updates
                (default is None).

        """
        super().__init__(model=model, data=data, aggregator=aggregator)

    def __get_xgb_model_data(self, train: bool = True) -> tuple[xgb.XGBModel, np.ndarray, np.ndarray]:
        """
        Get the XGBoost model and data arrays.

        Args:
            train (bool): Whether to get training data or test data
                (default is True).

        Returns:
            tuple[xgb.XGBModel, np.ndarray, np.ndarray]: A tuple containing
                (XGBoost model, features array, labels array).

        Raises:
            ValueError: If the model is not an XGBoost model.

        """
        # Get Model
        xgb_model = self.get_model().get_model()
        if not isinstance(xgb_model, xgb.XGBModel):
            raise ValueError("The model must be an XGBoost model")
        # Get Data
        X, y = self.get_data().export(XGBoostExportStrategy, train=train)
        return xgb_model, X, y

    @allow_no_addr_check
    def fit(self) -> P2PFLModel:
        """
        Fit the XGBoost sklearn model on training data.

        Trains the model for the configured number of epochs, continuing
        from a previous booster if available.

        Returns:
            P2PFLModel: The trained model with updated parameters.

        """
        model, X_train, y_train = self.__get_xgb_model_data(train=True)
        # prepare callbacks
        xgb_callbacks = []
        for cb in self.callbacks:
            # each P2PFLCallback should expose an XGBoost-compatible callback
            if hasattr(cb, "to_xgb_callback"):
                xgb_callbacks.append(cb.to_xgb_callback())

        try:
            # Try to get the booster from the current model to continue training
            previous_booster = self.get_model().get_model().get_booster()
            model.fit(
                X_train,
                y_train,
                verbose=True,
                xgb_model=previous_booster,  # Load previous model if exists
            )
        except (NotFittedError, Exception):
            # If no previous model exists or there's an error, start fresh
            model.fit(X_train, y_train, verbose=True)

        self.get_model().set_contribution([self.addr], self.get_data().get_num_samples(train=True))
        # store callback info back to model
        self.add_callback_info_to_model()
        return self.get_model()

    @allow_no_addr_check
    def interrupt_fit(self) -> None:
        """
        Interrupt an in-progress XGBoost training.

        Raises:
            NotImplementedError: XGBoost sklearn API does not support interruption.

        """
        # Placeholder: XGBoost sklearn does not support interrupt; could set a flag for custom callback
        raise NotImplementedError("Interrupting XGBoost sklearn fit is not supported")

    # def set_model(self, model: Union[P2PFLModel, list[np.ndarray], bytes]) -> None:
    #
    #     self.__model = model

    @allow_no_addr_check
    def evaluate(self) -> dict[str, float]:
        """
        Evaluate the model on test data.

        Computes accuracy and F1 score for classification tasks,
        or MSE for regression tasks.

        Returns:
            dict[str, float]: Dictionary containing evaluation metrics.
                For classification: 'accuracy' and 'f1'.
                For regression: 'mse'.

        """
        model, X_test, y_test = self.__get_xgb_model_data(train=False)
        results: dict[str, float] = {}
        try:
            preds = model.predict(X_test)
        except NotFittedError:
            return results

        # classification vs regression metric
        if np.issubdtype(y_test.dtype, np.integer):
            accuracy = float(np.mean(preds == y_test))
            results["accuracy"] = accuracy
            # Calcular F1 score
            f1 = f1_score(y_test, preds, average="weighted")
            results["f1"] = float(f1)
        else:
            mse = float(np.mean((preds - y_test) ** 2))
            results["mse"] = mse
        return results

    @allow_no_addr_check
    def get_framework(self) -> str:
        """
        Return the framework identifier.

        Returns:
            str: The framework name ('xgboost').

        """
        return Framework.XGBOOST.value
