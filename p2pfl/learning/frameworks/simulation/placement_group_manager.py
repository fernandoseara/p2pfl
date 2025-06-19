import threading

import ray
from ray.util.placement_group import placement_group, remove_placement_group


class PlacementGroupManager:
    """Singleton manager for a shared Ray placement group using all available resources by default."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, bundles: list[dict[str, float]] | None = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, bundles: list[dict[str, float]] | None = None):
        if self._initialized:
            return

        # If not provided, use all available resources in one bundle by default
        if bundles is None:
            available = ray.cluster_resources()
            # Filter out internal resources
            filtered = {
                k: v for k, v in available.items()
                if not k.startswith("node:") and not k.startswith("object_store")
            }

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

    def get_placement_group(self):
        return self.pg

    def get_bundle_count(self) -> int:
        return len(self.bundles)

    def shutdown(self):
        remove_placement_group(self.pg)
        self.__class__._instance = None
        self._initialized = False
