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

from dataclasses import dataclass
from typing import TYPE_CHECKING

from p2pfl.stages.network_state.network_state import NetworkState

if TYPE_CHECKING:
    from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


@dataclass
class PeerNodeState:
    """Class to store the state of a peer node."""

    round_number: int
    model_updated: P2PFLModel | None  # The model updated by this peer
    aggregated_from: list[str]  # Addresses of models this one was aggregated from
    train_set_votes: dict[str, int]  # for each nei the given vote


class BasicNetworkState(NetworkState):
    """Network state to keep track of peer nodes and their states."""

    def __init__(self) -> None:
        """Initialize the network state."""
        super().__init__()
        # address -> PeerNodeState
        self._peer_states: dict[str, PeerNodeState] = {}

    def add_peer(self, address: str) -> None:
        """Add a new peer to the network state."""
        if address not in self._peer_states:
            self._peer_states[address] = PeerNodeState(
                round_number=0, # Initial round number
                model_updated=None,
                aggregated_from=[],
                train_set_votes={},
            )
        else:
            raise ValueError(f"Address {address} already exists in network state.")

    def add_peer_state(self, address: str, state: PeerNodeState) -> None:
        """Add or update a peer's state."""
        self._peer_states[address] = state

    def add_model(self, model: P2PFLModel, source: str) -> None:
        """Add a model to a peer's state."""
        if source in self._peer_states:
            self._peer_states[source].model_updated = model
        else:
            raise ValueError(f"Address {source} not found in network state.")

    def add_vote(self, address: str, train_set_id: str, vote: int) -> None:
        """Add a vote to a peer's state."""
        if address in self._peer_states:
            if train_set_id not in self._peer_states[address].train_set_votes:
                self._peer_states[address].train_set_votes[train_set_id] = 0
            self._peer_states[address].train_set_votes[train_set_id] += vote
        else:
            raise ValueError(f"Address {address} not found in network state.")

    def add_aggregated_from(self, address: str, source: str) -> None:
        """Add a source to the aggregated_from list of a peer's state."""
        if address in self._peer_states:
            self._peer_states[address].aggregated_from.append(source)
        else:
            raise ValueError(f"Address {address} not found in network state.")

    def update_round(self, address: str, round_number: int) -> None:
        """Update the round number for a peer's state."""
        if address in self._peer_states:
            self._peer_states[address].round_number = round_number
        else:
            raise ValueError(f"Address {address} not found in network state.")

    def reset_round(self, address: str) -> None:
        """Reset the round number for a peer's state."""
        self.remove_model(address=address)
        self.clear_votes(address=address)
        self.clear_aggregated_from(address=address)

    def reset_all_rounds(self) -> None:
        """Reset the round number to 0 for all peers' states."""
        for address in self._peer_states:
            self.reset_round(address=address)

    def clear_aggregated_from(self, address: str) -> None:
        """Reset the aggregated model from for all peers' states."""
        if address in self._peer_states:
            self._peer_states[address].aggregated_from.clear()
        else:
            raise ValueError(f"Address {address} not found in network state.")

    def remove_model(self, address: str) -> None:
        """Remove a model from a peer's state."""
        if address in self._peer_states:
            self._peer_states[address].model_updated = None
        else:
            raise ValueError(f"Address {address} not found in network state.")

    def clear_votes(self, address: str) -> None:
        """Clear votes for a peer's state."""
        if address in self._peer_states:
            self._peer_states[address].train_set_votes.clear()
        else:
            raise ValueError(f"Address {address} not found in network state.")

    def clear_all_votes(self) -> None:
        """Clear all votes for all peers."""
        for state in self._peer_states.values():
            state.train_set_votes.clear()

    def get_peer_state(self, address: str) -> PeerNodeState | None:
        """Retrieve a peer's state by ID."""
        return self._peer_states.get(address)

    def get_model(self, address: str) -> P2PFLModel | None:
        """Retrieve the model updated by a specific peer."""
        state = self.get_peer_state(address)
        return state.model_updated if state else None

    def get_contributors(self, address: str) -> list[str] | None:
        """Retrieve the contributors of a specific peer's model."""
        state = self.get_peer_state(address)
        return state.model_updated.get_contributors() if state and state.model_updated else None

    def get_vote(self, address: str, train_set_id: str) -> int | None:
        """Retrieve the vote for a specific training set by a peer."""
        if address in self._peer_states:
            return self._peer_states[address].train_set_votes.get(train_set_id)
        else:
            raise ValueError(f"Address {address} not found in network state.")

    def get_votes(self, address: str) -> dict[str, int] | None:
        """Retrieve all votes for a specific peer."""
        state = self.get_peer_state(address)
        return state.train_set_votes if state else None

    def get_round(self, address: str) -> int | None:
        """Retrieve the peer round number."""
        state = self.get_peer_state(address)
        return state.round_number if state else None

    def get_all_models(self) -> list[P2PFLModel]:
        """Get all models currently stored."""
        return [state.model_updated for state in self._peer_states.values() if state.model_updated]

    def get_all_contributors(self) -> list[str]:
        """Get all contributors for the models."""
        contributors = []
        for state in self._peer_states.values():
            if state.model_updated:
                contributors.extend(state.model_updated.get_contributors())
        return list(set(contributors))  # Remove duplicates

    def get_all_votes(self) -> dict[str, dict[str, int]]:
        """Return all votes by peer, per training set."""
        return {address: state.train_set_votes for address, state in self._peer_states.items()}

    def get_all_rounds(self) -> dict[str, int]:
        """Return all rounds by peer."""
        return {address: state.round_number for address, state in self._peer_states.items()}

    def get_all_peers(self) -> dict[str, PeerNodeState]:
        """Return all peers and their states."""
        return self._peer_states

    def get_aggregation_sources(self, address: str) -> list[str] | None:
        """List addresses that contributed to a peer's aggregated model."""
        state = self.get_peer_state(address)
        return state.aggregated_from if state else None

    def get_peers_by_round(self, round_number: int) -> list[str]:
        """List Addresss that reported for a given round."""
        return [pid for pid, state in self._peer_states.items() if state.round_number == round_number]

    def remove_peer(self, address: str) -> None:
        """Remove a peer's state from the network."""
        if address in self._peer_states:
            del self._peer_states[address]

    def list_peers(self) -> list[str]:
        """List all Addresss in the network state."""
        return list(self._peer_states.keys())

    def clear(self) -> None:
        """Clear the network state."""
        self._peer_states.clear()

    def __len__(self) -> int:
        """Get the number of peers in the network state."""
        return len(self._peer_states)

    def __contains__(self, address: str) -> bool:
        """Check if a Address is in the network state."""
        return address in self._peer_states

    def __repr__(self) -> str:
        """Representation of the network state."""
        return f"<NetworkState: {len(self._peer_states)} peer nodes>"
