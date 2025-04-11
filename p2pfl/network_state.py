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
"""Netowork state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel

@dataclass
class PeerNodeState:
    """Class to store the state of a peer node."""

    round_number: int
    model_updated: P2PFLModel # The model updated by this peer
    aggregated_from: list[str] # Addresses of models this one was aggregated from
    train_set_votes: dict[str, int] # for each nei the given vote

class NetworkState:
    """Network state to keep track of peer nodes and their states."""

    def __init__(self):
        """Initialize the network state."""
        # peer_id -> PeerNodeState
        self._peer_states: dict[str, PeerNodeState] = {}

    def add_peer_state(self, peer_id: str, state: PeerNodeState):
        """Add or update a peer's state."""
        self._peer_states[peer_id] = state

    def add_model(self, model: P2PFLModel, source: str):
        """Add a model to a peer's state."""
        if source in self._peer_states:
            self._peer_states[source].model_updated = model
            self._peer_states[source].aggregated_from.append(source)
        else:
            raise ValueError(f"Peer ID {source} not found in network state.")

    def add_vote(self, peer_id: str, train_set_id: str, vote: int):
        """Add a vote to a peer's state."""
        if peer_id in self._peer_states:
            if train_set_id not in self._peer_states[peer_id].train_set_votes:
                self._peer_states[peer_id].train_set_votes[train_set_id] = 0
            self._peer_states[peer_id].train_set_votes[train_set_id] += vote
        else:
            raise ValueError(f"Peer ID {peer_id} not found in network state.")

    def update_round(self, peer_id: str, round_number: int):
        """Update the round number for a peer's state."""
        if peer_id in self._peer_states:
            self._peer_states[peer_id].round_number = round_number
        else:
            raise ValueError(f"Peer ID {peer_id} not found in network state.")

    def reset_round(self, peer_id): 
        """Reset the round number to 0 for all peers' states."""
        self.remove_model(peer_id=peer_id)
        self.clear_votes(peer_id=peer_id)
        self.clear_aggregated_from(peer_id=peer_id)

    def clear_aggregated_from(self, peer_id):
        """Reset the aggregated model from for all peers' states."""
        if peer_id in self._peer_states:
            self._peer_states[peer_id].aggregated_from = None
        else:
            raise ValueError(f"Peer ID {peer_id} not found in network state.")

    def remove_model(self, peer_id: str):
        """Remove a model from a peer's state."""
        if peer_id in self._peer_states:
            self._peer_states[peer_id].model_updated = None
        else:
            raise ValueError(f"Peer ID {peer_id} not found in network state.")

    def clear_votes(self, peer_id: str):
        """Clear votes for a peer's state."""
        if peer_id in self._peer_states:
            self._peer_states[peer_id].train_set_votes.clear()
        else:
            raise ValueError(f"Peer ID {peer_id} not found in network state.")

    def clear_all_votes(self):
        """Clear all votes for all peers."""
        for state in self._peer_states.values():
            state.train_set_votes.clear()

    def get_peer_state(self, peer_id: str) -> PeerNodeState | None:
        """Retrieve a peer's state by ID."""
        return self._peer_states.get(peer_id)

    def get_model(self, peer_id: str) -> P2PFLModel | None:
        """Retrieve the model updated by a specific peer."""
        state = self.get_peer_state(peer_id)
        return state.model_updated if state else None

    def get_vote(self, peer_id: str, train_set_id: str) -> int | None:
        """Retrieve the vote for a specific training set by a peer."""
        if peer_id in self._peer_states:
            return self._peer_states[peer_id].train_set_votes.get(train_set_id)
        else:
            raise ValueError(f"Peer ID {peer_id} not found in network state.")


    def get_all_models(self) -> list[P2PFLModel]:
        """Get all models currently stored."""
        return [state.model_updated for state in self._peer_states.values()]

    def get_all_votes(self) -> dict[str, dict[str, int]]:
        """Return all votes by peer, per training set."""
        return {peer_id: state.train_set_votes for peer_id, state in self._peer_states.items()}

    def get_aggregation_sources(self, peer_id: str) -> list[str] | None:
        """List addresses that contributed to a peer's aggregated model."""
        state = self.get_peer_state(peer_id)
        return state.aggregated_from if state else None

    def get_peers_by_round(self, round_number: int) -> list[str]:
        """List peer IDs that reported for a given round."""
        return [pid for pid, state in self._peer_states.items() if state.round_number == round_number]

    def remove_peer(self, peer_id: str):
        """Remove a peer's state from the network."""
        if peer_id in self._peer_states:
            del self._peer_states[peer_id]

    def list_peers(self) -> list[str]:
        """List all peer IDs in the network state."""
        return list(self._peer_states.keys())

    def clear(self):
        """Clear the network state."""
        self._peer_states.clear()

    def __len__(self):
        """Get the number of peers in the network state."""
        return len(self._peer_states)

    def __contains__(self, peer_id: str):
        """Check if a peer ID is in the network state."""
        return peer_id in self._peer_states

    def __repr__(self):
        """Representation of the network state."""
        return f"<NetworkState: {len(self._peer_states)} peer nodes>"
