#
# This file is part of the p2pfl distribution (see https://github.com/pguijas/p2pfl).
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
"""
Workflow commands module.

These commands dynamically route messages to the active learning workflow.
They are registered when a workflow starts and removed when it stops.

- MessageCommand: Routes gossip/direct messages (string args) to `on_message_<name>` handlers
- WeightsCommand: Routes weight transfers (binary data) to `on_weights_<name>` handlers
"""

from p2pfl.communication.commands.workflow.message_command import MessageCommand
from p2pfl.communication.commands.workflow.weights_command import WeightsCommand

__all__ = [
    "MessageCommand",
    "WeightsCommand",
]
