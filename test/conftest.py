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

"""Pytest configuration for p2pfl tests."""

import contextlib
import importlib.util
import os
import sys

# Disable loading ~/.p2pfl_env during tests to prevent interference
os.environ["P2PFL_TESTING"] = "1"

#
# MacOS Problem FIXES
#

if sys.platform == "darwin":
    os.environ.setdefault("no_proxy", "*")
    os.environ.setdefault("NO_PROXY", "*")

with contextlib.suppress(ImportError):
    import torch  # noqa: F401

with contextlib.suppress(ImportError):
    import tensorflow  # noqa: F401

#
# Ray initialization: import logger first (may init Ray), then disable for tests
#

import pytest

from p2pfl.management.logger import logger  # noqa: F401, E402
from p2pfl.settings import Settings

# Disable Ray for all tests by default
Settings.general.DISABLE_RAY = True


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "uses_ray: mark test as requiring Ray",
    )
    config.addinivalue_line(
        "markers",
        "e2e_train: end-to-end training tests (skip with -m 'not e2e_train')",
    )


@pytest.fixture(autouse=True)
def handle_ray_marker(request):
    """Enable Ray for tests marked with @pytest.mark.uses_ray."""
    if request.node.get_closest_marker("uses_ray"):
        # Skip if Ray is not installed
        if importlib.util.find_spec("ray") is None:
            pytest.skip("Ray is not installed")

        Settings.general.DISABLE_RAY = False
        yield
        Settings.general.DISABLE_RAY = True
    else:
        yield
