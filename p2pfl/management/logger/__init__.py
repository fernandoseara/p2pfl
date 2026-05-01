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

"""Provides a logger singleton that can be used to log messages from different parts of the codebase."""

from p2pfl.management.logger.decorators.async_logger import AsyncLogger
from p2pfl.management.logger.decorators.file_logger import FileLogger
from p2pfl.management.logger.decorators.singleton_logger import SingletonLogger
from p2pfl.management.logger.decorators.wandb_logger import WandbLogger
from p2pfl.management.logger.decorators.web_logger import WebP2PFLogger
from p2pfl.management.logger.logger import P2PFLogger

__all__ = ["logger", "P2PFLogger", "AsyncLogger", "FileLogger", "SingletonLogger", "WandbLogger", "WebP2PFLogger"]

# Module-level type annotation for mypy
logger: P2PFLogger

# This is only executed once, when the module is first imported
logger = SingletonLogger(AsyncLogger(WandbLogger(WebP2PFLogger(FileLogger(P2PFLogger(disable_locks=False))))))
