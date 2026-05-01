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
"""YAML launcher utilities."""

import importlib
import json
import os
import random
from typing import Any

import pandas as pd
from datasets import DatasetDict

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.settings import Settings


def load_by_package_and_name(package_name: str, class_name: str) -> Any:
    """Load a class by package and name."""
    module = importlib.import_module(package_name)
    return getattr(module, class_name)


def resize_partitions(
    partitions: list[P2PFLDataset],
    samples_per_node: int,
    test_samples_per_node: int | None = None,
) -> list[P2PFLDataset]:
    """Resize partitions to a fixed number of samples per node (for scalability tests)."""
    rng = random.Random(Settings.general.SEED)

    def _resize(data: Any, target_size: int) -> Any:
        current_size = len(data)
        if current_size == target_size:
            return data
        elif current_size > target_size:
            return data.select(list(range(target_size)))
        else:
            indices = list(range(current_size))
            indices += rng.choices(range(current_size), k=target_size - current_size)
            return data.select(indices)

    result = []
    for p in partitions:
        train_split = p._train_split_name
        test_split = p._test_split_name
        train_data = p._data[train_split]
        test_data = p._data[test_split]

        new_train = _resize(train_data, samples_per_node)

        if test_samples_per_node is not None:
            test_target = test_samples_per_node
        else:
            ratio = len(test_data) / max(len(train_data), 1)
            test_target = max(1, round(samples_per_node * ratio))
        new_test = _resize(test_data, test_target)

        result.append(
            P2PFLDataset(
                DatasetDict({train_split: new_train, test_split: new_test}),
                train_split_name=train_split,
                test_split_name=test_split,
                batch_size=p.batch_size,
                dataset_name=p.dataset_name,
            )
        )
    return result


def export_experiment_data(
    exp_dir: str,
    all_timings: list[dict[str, Any]],
    messages: list[Any],
    global_metrics: dict[str, Any] | None,
) -> None:
    """Export experiment results (timings, communication logs, metrics) to disk."""
    os.makedirs(exp_dir, exist_ok=True)

    if all_timings:
        with open(os.path.join(exp_dir, "stage_timings.json"), "w") as f:
            json.dump(all_timings, f, indent=2)

    if messages:
        with open(os.path.join(exp_dir, "communication_logs.json"), "w") as f:
            json.dump(messages, f, indent=2, default=str)

    if global_metrics:
        flat = []
        for exp, exp_nodes in global_metrics.items():
            for node_addr, metrics in exp_nodes.items():
                for metric_name, values in metrics.items():
                    for round_num, value in values:
                        flat.append(
                            {
                                "experiment": exp,
                                "node": node_addr,
                                "metric": metric_name,
                                "round": round_num,
                                "value": value,
                            }
                        )
        if flat:
            pd.DataFrame(flat).to_csv(os.path.join(exp_dir, "global_metrics.csv"), index=False)
