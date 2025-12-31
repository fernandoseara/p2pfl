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

"""Tests to verify the @pytest.mark.uses_ray marker and Ray initialization."""

import pytest


class TestRayMarker:
    """
    Test that Ray can be enabled/disabled via the uses_ray marker.

    The @pytest.mark.uses_ray marker automatically sets Settings.general.DISABLE_RAY = False
    for the duration of the test.
    """

    def test_ray_disabled_by_default(self):
        """Verify Ray is disabled by default in tests via Settings."""
        from p2pfl.settings import Settings
        from p2pfl.utils.check_ray import ray_installed

        assert Settings.general.DISABLE_RAY is True, "DISABLE_RAY should be True by default in tests"
        assert ray_installed() is False, "ray_installed() should return False when DISABLE_RAY=True"

    @pytest.mark.uses_ray
    def test_ray_enabled_with_marker(self):
        """Verify Ray can be enabled using the @pytest.mark.uses_ray marker."""
        from p2pfl.settings import Settings
        from p2pfl.utils.check_ray import ray_installed

        assert Settings.general.DISABLE_RAY is False, "DISABLE_RAY should be False with uses_ray marker"
        assert ray_installed() is True, "ray_installed() should return True with uses_ray marker"

        # Verify Ray is actually initialized
        import ray

        assert ray.is_initialized(), "Ray should be initialized"

    def test_ray_remains_disabled_after_uses_ray_test(self):
        """Verify Ray is disabled again after a uses_ray test completes."""
        from p2pfl.settings import Settings
        from p2pfl.utils.check_ray import ray_installed

        assert Settings.general.DISABLE_RAY is True, "DISABLE_RAY should be True after uses_ray test"
        assert ray_installed() is False, "ray_installed() should return False in normal tests"


class TestRayLogger:
    """Test that logger initializes with Ray when Ray is available."""

    def test_logger_uses_ray_when_available(self):
        """Verify logger is RayP2PFLogger when Ray is installed."""
        from p2pfl.management.logger import logger
        from p2pfl.management.logger.decorators.ray_logger import RayP2PFLogger

        # Logger should be Ray-based since Ray was initialized at import time
        assert isinstance(logger, RayP2PFLogger), f"Expected RayP2PFLogger, got {type(logger).__name__}"

    def test_ray_initialized_at_import(self):
        """Verify Ray is initialized (was initialized when logger was imported)."""
        import ray

        assert ray.is_initialized(), "Ray should be initialized at conftest import time"

    def test_ray_installed_respects_disable_setting(self):
        """Verify ray_installed() respects Settings.general.DISABLE_RAY."""
        from p2pfl.settings import Settings
        from p2pfl.utils.check_ray import ray_installed

        # Default in tests: disabled
        assert Settings.general.DISABLE_RAY is True
        assert ray_installed() is False

        # Enable
        Settings.general.DISABLE_RAY = False
        assert ray_installed() is True

        # Disable again
        Settings.general.DISABLE_RAY = True
        assert ray_installed() is False


class TestLearnerWrapping:
    """Test that learner wrapping respects Ray enable/disable setting."""

    def test_learner_not_wrapped_when_ray_disabled(self):
        """Verify learner is NOT wrapped in VirtualNodeLearner when Ray is disabled."""
        from p2pfl.learning.frameworks.pytorch.lightning_learner import LightningLearner
        from p2pfl.learning.frameworks.simulation import try_init_learner_with_ray
        from p2pfl.learning.frameworks.simulation.virtual_learner import VirtualNodeLearner
        from p2pfl.settings import Settings

        assert Settings.general.DISABLE_RAY is True

        learner = LightningLearner()
        wrapped = try_init_learner_with_ray(learner)

        assert wrapped is learner, "Learner should NOT be wrapped when Ray is disabled"
        assert not isinstance(wrapped, VirtualNodeLearner)

    @pytest.mark.uses_ray
    def test_learner_wrapped_when_ray_enabled(self):
        """Verify learner IS wrapped in VirtualNodeLearner when Ray is enabled."""
        from p2pfl.learning.frameworks.pytorch.lightning_learner import LightningLearner
        from p2pfl.learning.frameworks.simulation import try_init_learner_with_ray
        from p2pfl.learning.frameworks.simulation.virtual_learner import VirtualNodeLearner
        from p2pfl.settings import Settings

        assert Settings.general.DISABLE_RAY is False

        learner = LightningLearner()
        wrapped = try_init_learner_with_ray(learner)

        assert isinstance(wrapped, VirtualNodeLearner), f"Expected VirtualNodeLearner, got {type(wrapped).__name__}"
        assert wrapped.learner is learner
