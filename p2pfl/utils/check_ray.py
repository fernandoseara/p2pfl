#
# This file is part of the p2pfl distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2025 Pedro Guijas Bravo.
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

"""Check if ray is installed."""

import importlib
import os
import sys

from p2pfl.settings import Settings


def _worker_setup() -> None:
    """Import ML frameworks first in Ray workers to avoid deadlocks on macOS."""
    if sys.platform != "darwin":
        return
    import contextlib

    print("[p2pfl] Ray worker setup: pre-importing ML frameworks to avoid macOS deadlocks...")
    with contextlib.suppress(ImportError):
        import torch  # noqa: F401

        print(f"[p2pfl] Imported torch {torch.__version__}")
    with contextlib.suppress(ImportError):
        import tensorflow  # noqa: F401

        print(f"[p2pfl] Imported tensorflow {tensorflow.__version__}")


def ray_installed() -> bool:
    """Check if ray is installed."""
    if os.environ.get("P2PFL_DISABLE_RAY") or Settings.general.DISABLE_RAY:
        return False

    os.environ["RAY_DEDUP_LOGS"] = "0"

    if importlib.util.find_spec("ray") is not None:
        # Try to initialize ray
        import ray

        # If ray not initialized, initialize it
        if not ray.is_initialized():
            init_kwargs = {
                "namespace": "p2pfl",
                "include_dashboard": False,
                "logging_level": Settings.general.LOG_LEVEL,
                "logging_config": ray.LoggingConfig(encoding="TEXT", log_level=Settings.general.LOG_LEVEL),
            }
            # macOS: Import TF/Torch first in workers to avoid deadlocks with HuggingFace to_tf_dataset()
            if sys.platform == "darwin":
                init_kwargs["runtime_env"] = {"worker_process_setup_hook": _worker_setup}
            ray.init(**init_kwargs)
        return True
    return False
