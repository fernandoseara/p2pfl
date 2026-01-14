#
# This file is part of the p2pfl distribution
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

"""FedProx Aggregator - Federated Proximal."""

from p2pfl.learning.aggregators.fedavg import FedAvg
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


class FedProx(FedAvg):
    """
    FedProx - Federated Proximal [Li et al., 2018].

    Inherits from ``FedAvg`` (which inherits from ``WeightAggregator``)
    as FedProx works with neural network weight tensors.

    FedProx extends FedAvg by adding a proximal term to the local objective
    function to handle system and statistical heterogeneity.

    Note:
        Requires the ``fedprox`` callback (e.g., ``FedProxCallback`` for PyTorch).

        - Aggregator passes ``proximal_mu`` via ``additional_info``
        - Callback snapshots model params at training start as the global reference
        - During training, callback adds proximal penalty: ``μ × (w - w_global)``

        Paper: https://arxiv.org/abs/1812.06127

    """

    def __init__(self, proximal_mu: float = 0.01, disable_partial_aggregation: bool = False) -> None:
        """
        Initialize the FedProx aggregator.

        Args:
            proximal_mu: The proximal coefficient (μ) that controls the strength
                        of the regularization. Defaults to 0.01.
            disable_partial_aggregation: Whether to disable partial aggregation.

        """
        super().__init__(disable_partial_aggregation=disable_partial_aggregation)
        self.proximal_mu = proximal_mu

    def _aggregate(self, models: list[P2PFLModel]) -> P2PFLModel:
        """
        Aggregate models using FedAvg and pass the proximal coefficient.

        Args:
            models: List of models to aggregate.

        Returns:
            A P2PFLModel with the aggregated parameters and proximal coefficient.

        """
        # Use FedAvg aggregation
        aggregated_model = super()._aggregate(models)

        # Set the proximal_mu for the callback
        # Note: initial params are captured by the callback at training start
        aggregated_model.add_info("fedprox", {"proximal_mu": self.proximal_mu})

        return aggregated_model

    def get_required_callbacks(self) -> list[str]:
        """
        Get the required callbacks for FedProx.

        Returns:
            List of required callbacks.

        """
        return ["fedprox"]
