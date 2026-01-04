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

    round_number: int # The index of the local iteration of the peer (tl_j,i)
    push_sum_weight: float # The weight of the push sum algorithm (μ)
    model_updated: P2PFLModel|None # The model updated by this peer (w)
    losses: list[float] # The loss of the model updated by this peer in all rounds (f_i(ω))
    push_time: int # The last model push time (tp_j,i)
    mixing_weight: float # Mixing weights for the push sum algorithm (pt_j,i)
    p2p_updating_idx: int # Index of the latest P2P updating process (t_j)

class AsyncNetworkState(NetworkState):
    """Network state to keep track of peer nodes and their states."""

    def __init__(self) -> None:
        """Initialize the network state."""
        # address -> PeerNodeState
        self._peer_states: dict[str, PeerNodeState] = {}

    def add_peer(self, address: str) -> None:
        """Add a new peer to the network state."""
        if address not in self._peer_states:
            self._peer_states[address] = PeerNodeState(
                round_number=0,
                push_sum_weight=1.0,        # μ(0)
                model_updated=None,         # w(0)
                losses=[],                  # f_i(ω(0))
                push_time=0,                # tp_j,i(0)
                mixing_weight=1.0,          # pt_j,i(0)
                p2p_updating_idx=0,         # t_j(0)
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

    def add_loss(self, address: str, round: int, loss: float) -> None:
        """Add a loss to a peer's state."""
        if address in self._peer_states:
            self._peer_states[address].losses.insert(round, loss)
        else:
            raise ValueError(f"Address {address} not found in network state.")

    def update_push_time(self, address: str, time: int) -> None:
        """Add the last push time for a peer's state."""
        if address in self._peer_states:
            self._peer_states[address].push_time = time
        else:
            raise ValueError(f"Address {address} not found in network state.")

    def update_p2p_updating_idx(self, address: str, idx: int) -> None:
        """Add the P2P updating index for a peer's state."""
        if address in self._peer_states:
            self._peer_states[address].p2p_updating_idx = idx
        else:
            raise ValueError(f"Address {address} not found in network state.")

    def update_round(self, address: str, round_number: int) -> None:
        """Update the round number for a peer's state."""
        if address in self._peer_states:
            self._peer_states[address].round_number = round_number
        else:
            raise ValueError(f"Address {address} not found in network state.")

    def update_push_sum_weight(self, address: str, weight: float) -> None:
        """Update the push sum weight for a peer's state."""
        if address in self._peer_states:
            self._peer_states[address].push_sum_weight = weight
        else:
            raise ValueError(f"Address {address} not found in network state.")

    def update_mixing_weight(self, address: str, mixing_weight: float) -> None:
        """Update the mixing weight for a peer's state."""
        if address in self._peer_states:
            self._peer_states[address].mixing_weight = mixing_weight
        else:
            raise ValueError(f"Address {address} not found in network state.")

    def set_mixing_weights(self, mixing_weights: dict[str, float]) -> None:
        """Update the mixing weights for all peers."""
        for address, weights in mixing_weights.items():
            if address in self._peer_states:
                self._peer_states[address].mixing_weight = weights
            else:
                raise ValueError(f"Address {address} not found in network state.")

    def set_push_times(self, push_times: dict[str, int]) -> None:
        """Update the push times for all peers."""
        for address, time in push_times.items():
            if address in self._peer_states:
                self._peer_states[address].push_time = time
            else:
                raise ValueError(f"Address {address} not found in network state.")

    def reset_round(self, address: str) -> None:
        """Reset the round number for a peer's state."""
        if address in self._peer_states:
            self._peer_states[address].round_number = 0
        else:
            raise ValueError(f"Address {address} not found in network state.")

    def remove_model(self, address: str) -> None:
        """Remove a model from a peer's state."""
        if address in self._peer_states:
            self._peer_states[address].model_updated = None
        else:
            raise ValueError(f"Address {address} not found in network state.")

    def clear_all_models(self) -> None:
        """Clear all models from the network state."""
        for address in self._peer_states:
            self.remove_model(address=address)

    def clear_losses(self, address: str) -> None:
        """Clear the losses of a peer's state."""
        if address in self._peer_states:
            self._peer_states[address].losses = []
        else:
            raise ValueError(f"Address {address} not found in network state.")

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

    def get_round(self, address: str) -> int | None:
        """Retrieve the peer round number."""
        state = self.get_peer_state(address)
        return state.round_number if state else None

    def get_push_sum_weight(self, address: str) -> float | None:
        """Retrieve the push sum weight of a specific peer."""
        state = self.get_peer_state(address)
        return state.push_sum_weight if state else None

    def get_push_time(self, address: str) -> int | None:
        """Retrieve the last push time of a specific peer."""
        state = self.get_peer_state(address)
        return state.push_time if state else None

    def get_p2p_updating_idx(self, address: str) -> int | None:
        """Retrieve the P2P updating index of a specific peer."""
        state = self.get_peer_state(address)
        return state.p2p_updating_idx if state else None

    def get_mixing_weight(self, address: str) -> float | None:
        """Retrieve the mixing weight of a specific peer."""
        state = self.get_peer_state(address)
        return state.mixing_weight if state else None

    def get_losses(self, address: str) -> list[float] | None:
        """Retrieve the losses of a specific peer."""
        state = self.get_peer_state(address)
        return state.losses if state else None

    def get_all_models(self) -> list[P2PFLModel]:
        """Get all models currently stored."""
        return [state.model_updated for state in self._peer_states.values() if state.model_updated]

    def get_all_contributors(self) -> list[str]:
        """Get all contributors for the models."""
        contributors = []
        for state in self._peer_states.values():
            if state.model_updated:
                contributors.extend(state.model_updated.get_contributors())
        return list(set(contributors)) # Remove duplicates

    def get_all_rounds(self) -> dict[str, int]:
        """Return all rounds by peer."""
        return {address: state.round_number for address, state in self._peer_states.items()}

    def get_all_peers(self) -> dict[str, PeerNodeState]:
        """Return all peers and their states."""
        return self._peer_states

    def get_all_push_sum_weights(self) -> dict[str, float]:
        """Return all push sum weights by peer."""
        return {address: state.push_sum_weight for address, state in self._peer_states.items()}

    def get_all_push_times(self) -> dict[str, int]:
        """Return all push times by peer."""
        return {address: state.push_time for address, state in self._peer_states.items()}

    def get_all_mixing_weights(self) -> dict[str, float]:
        """Return all mixing weights by peer."""
        return {address: state.mixing_weight for address, state in self._peer_states.items()}

    def get_all_losses(self) -> dict[str, list[float]]:
        """Return all losses by peer."""
        return {address: state.losses for address, state in self._peer_states.items()}

    def get_peers_by_round(self, round_number: int) -> list[str]:
        """List Addresss that reported for a given round."""
        return [pid for pid, state in self._peer_states.items() if state.round_number == round_number]

    def remove_peer(self, address: str):
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
