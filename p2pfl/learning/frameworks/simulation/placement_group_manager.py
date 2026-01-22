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

"""Manage Ray placement groups."""

import threading

import ray
from ray.util.placement_group import PlacementGroup, placement_group, remove_placement_group


class PlacementGroupManager:
    """Singleton manager for a shared Ray placement group using all available resources by default."""

    _instance: "PlacementGroupManager | None" = None
    _lock = threading.Lock()
    _initialized: bool

    def __new__(cls, bundles: list[dict[str, float]] | None = None) -> "PlacementGroupManager":
        """
        Create or return the singleton instance.

        Args:
            bundles: List of resource bundles for the placement group. If None, use all available resources
                        in one or more bundles.

        Returns:
            The singleton PlacementGroupManager instance.

        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, bundles: list[dict[str, float]] | None = None) -> None:
        """
        Initialize the placement group manager.

        Args:
            bundles: List of resource bundles for the placement group. If None, use all available resources
                        in one or more bundles.

        """
        if self._initialized:
            return

        # If not provided, use all available resources in one bundle by default
        if bundles is None:
            available = ray.cluster_resources()
            # Filter out internal resources
            filtered = {k: v for k, v in available.items() if not k.startswith("node:") and not k.startswith("object_store")}

            bundles = []
            cpu = filtered.pop("CPU", 0)
            gpu = filtered.pop("GPU", 0)

            # If GPUs exist, put CPUs and GPUs together
            if gpu > 0:
                bundles.append({"CPU": cpu, "GPU": gpu})
            else:
                if cpu > 0:
                    bundles.append({"CPU": cpu})

            # Add remaining resources as individual bundles
            for res, val in filtered.items():
                bundles.append({res: val})

        self.bundles = bundles
        self.pg = placement_group(self.bundles, strategy="PACK")
        ray.get(self.pg.ready())
        self._initialized = True

    def get_placement_group(self) -> PlacementGroup:
        """
        Get the Ray placement group.

        Returns:
            The Ray placement group.

        """
        return self.pg

    def get_bundle_count(self) -> int:
        """Get the number of bundles in the placement group."""
        return len(self.bundles)

    def shutdown(self) -> None:
        """Remove the placement group and reset the singleton instance."""
        remove_placement_group(self.pg)
        self.__class__._instance = None
        self._initialized = False
