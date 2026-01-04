#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2024 Pedro Guijas Bravo.
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
"""Network state."""

from abc import ABC, abstractmethod


class NetworkState(ABC):
    """Network state to keep track of peer nodes and their states."""

    def __init__(self) -> None:
        """Initialize the network state."""
        self.is_initiator: bool = False

    def check_is_initiator(self) -> bool:
        """Check if the node is the initiator of the network."""
        return self.is_initiator

    def set_initiator(self, is_initiator: bool) -> None:
        """Set the node as the initiator of the network."""
        self.is_initiator = is_initiator

    @abstractmethod
    def add_peer(self, address: str):
        """Add a new peer to the network state."""
        pass
