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

"""Push-Sum Aggregator for directed communication topologies."""

import numpy as np

from p2pfl.learning.aggregators.aggregator import NoModelsToAggregateError, WeightAggregator
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


class PushSum(WeightAggregator):
    """
    Push-Sum aggregator (Eq. 5 & 6 from AsyDFL paper).

    Similar to FedAvg but with two key differences:

    - **Weight source**: uses topology mixing weights (``mixing_weight``)
      instead of sample counts.
    - **No normalization**: the weighted sum is *not* divided by the total
      weight.  The push-sum protocol corrects for this externally by
      dividing the accumulated model by the push-sum scalar μ.

    Each model must carry two info fields set via ``add_info()``:

    - ``mixing_weight`` (float): the mixing coefficient p_{i,j}.
    - ``push_sum_weight`` (float): the push-sum scalar μ_j.

    The aggregated model will have ``push_sum_weight`` set to the new μ
    (Eq. 6: μ_i = Σ p_{i,j} · μ_j).
    """

    def _aggregate(self, models: list[P2PFLModel]) -> P2PFLModel:
        """Aggregate models (Eq. 5) and push-sum weights (Eq. 6)."""
        if len(models) == 0:
            raise NoModelsToAggregateError(f"({self.address}) Trying to aggregate models when there is no models")

        first_params = models[0].get_parameters()
        accum = [np.zeros_like(layer) for layer in first_params]

        # Eq. (6): μ_i^{t+1} = Σ p_{i,j} · μ_j
        push_sum_weight = 0.0
        contributors: list[str] = []
        total_samples = 0
        for m in models:
            mixing = m.get_info().get("mixing_weight", 1.0)
            mu = m.get_info().get("push_sum_weight", 1.0)
            push_sum_weight += mixing * mu
            # Eq. (5): ω_i^{t+1} = Σ p_{i,j} · ω_j
            for i, layer in enumerate(m.get_parameters()):
                accum[i] = np.add(accum[i], layer * mixing)
            contributors = contributors + m.get_contributors()
            total_samples += m.get_num_samples()

        result = models[0].build_copy(params=accum, num_samples=total_samples, contributors=contributors)
        result.add_info("push_sum_weight", push_sum_weight)
        return result
