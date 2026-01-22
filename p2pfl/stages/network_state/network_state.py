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

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


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

    # Core peer management - abstract methods
    @abstractmethod
    def add_peer(self, address: str) -> None:
        """Add a new peer to the network state."""
        pass

    @abstractmethod
    def list_peers(self) -> list[str]:
        """List all addresses in the network state."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear the network state."""
        pass

    # Common model operations - abstract methods
    @abstractmethod
    def add_model(self, model: P2PFLModel, source: str) -> None:
        """Add a model to a peer's state."""
        pass

    @abstractmethod
    def get_all_models(self) -> list[P2PFLModel]:
        """Get all models currently stored."""
        pass

    @abstractmethod
    def get_all_contributors(self) -> list[str]:
        """Get all contributors for the models."""
        pass

    # Round management - abstract methods
    @abstractmethod
    def update_round(self, address: str, round_number: int) -> None:
        """Update the round number for a peer's state."""
        pass

    @abstractmethod
    def get_round(self, address: str) -> int | None:
        """Retrieve the peer round number."""
        pass
