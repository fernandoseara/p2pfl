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

"""XGBoost DMatrix export integration."""

import numpy as np
from datasets import Dataset

from p2pfl.learning.dataset.p2pfl_dataset import DataExportStrategy


class XGBoostExportStrategy(DataExportStrategy):
    """
    Export strategy for XGBoost datasets.

    Converts P2PFL datasets to numpy arrays suitable for XGBoost training.

    """

    @staticmethod
    def export(data: Dataset, batch_size: int | None = None, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Export dataset to numpy arrays for XGBoost.

        Args:
            data (Dataset): The dataset to export.
            batch_size (int | None): The batch size for the export (unused for XGBoost).
            **kwargs: Additional keyword arguments:
                - train (bool): Whether this is training data (default is True).
                - label_key (str | None): The key for the label column
                    (default is None, uses last column).
                - feature_keys (list[str] | None): List of feature column keys
                    (default is None, uses all except label).

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple of (features, labels) as numpy arrays.

        """
        # Extract XGBoost-specific parameters from kwargs
        label_key = kwargs.get("label_key")
        feature_keys = kwargs.get("feature_keys")

        # Convert to pandas and then numpy
        df = data.to_pandas()
        if label_key is None:
            label_key = df.columns[-1]
        keys = feature_keys or [c for c in df.columns if c != label_key]
        X = df[keys].to_numpy()
        y = df[label_key].to_numpy()
        return X, y
