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

"""Communication protocol."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

from p2pfl.communication.commands.command import Command
from p2pfl.utils.node_component import NodeComponent, allow_no_addr_check


class CommunicationProtocol(ABC, NodeComponent):
    """
    Communication protocol interface.

    Args:
        commands: The commands.

    """

    def __init__(self, commands: Optional[List[Command]] = None, *args, **kwargs) -> None:
        """Initialize the communication protocol."""
        # (addr) Super
        NodeComponent.__init__(self)

    @abstractmethod
    async def start(self) -> None:
        """Start the communication protocol."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the communication protocol."""
        pass

    @allow_no_addr_check
    @abstractmethod
    def add_command(self, cmds: Union[Command, List[Command]]) -> None:
        """
        Add a command to the communication protocol.

        Args:
            cmds: The command to add.

        """
        pass

    @allow_no_addr_check
    @abstractmethod
    def remove_command(self, cmd: Union[str, Command]) -> None:
        """
        Remove a command from the communication protocol.

        Args:
            cmd: The command to remove.

        """
        pass

    @abstractmethod
    def build_msg(self, command_name: str, content: Optional[list[Any]] = None, round: Optional[int] = None) -> Any:
        """
        Build a message to send to the neighbors.

        Args:
            command_name: Command of the message.
            content: Content of the message.
            round: Round of the message.

        Returns:
            Message to send.

        """
        pass

    @abstractmethod
    def build_weights(
        self,
        cmd: str,
        round: int,
        serialized_model: bytes,
        contributors: Optional[list[str]] = None,
        weight: int = 1,
    ) -> Any:
        """
        Build weights.

        Args:
            cmd: The command.
            round: The round.
            serialized_model: The serialized model.
            contributors: The model contributors.
            weight: The weight of the model (amount of samples used).

        """
        pass

    @abstractmethod
    async def send(
        self,
        nei: str,
        msg: Any,
        raise_error: bool = False,
        remove_on_error: bool = True,
        temporal_connection: bool = False,
    ) -> None:
        """
        Send a message to a neighbor.

        Args:
            nei: The neighbor to send the message.
            msg: The message to send.
            raise_error: If raise error.
            remove_on_error: If remove on error.

        """
        pass

    @abstractmethod
    async def broadcast(self, msg: Any, node_list: Optional[List[str]] = None) -> None:
        """
        Broadcast a message to all neighbors.

        Args:
            msg: The message to broadcast.
            node_list: Optional node list.

        """
        pass

    @abstractmethod
    async def gossip(
        self,
        nei: str,
        msg: Any,
        raise_error: bool = False,
        remove_on_error: bool = True,
        temporal_connection: bool = False,
    ) -> None:
        """
        Gossip a message to a neighbor.

        Args:
            nei: The neighbor to gossip the message.
            msg: The message to gossip.
            raise_error: If raise error.
            remove_on_error: If remove on error.

        """
        pass

    @abstractmethod
    async def broadcast_gossip(self, msg: Any, node_list: Optional[List[str]] = None) -> None:
        """
        Gossip a message to all neighbors.

        Args:
            msg: The message to gossip.
            node_list: Optional node list.

        """
        pass

    @abstractmethod
    async def connect(self, address: str, non_direct: bool = False) -> bool:
        """
        Connect to a neighbor.

        Args:
            address: The address to connect to.
            non_direct: The non direct flag.

        """
        pass

    @abstractmethod
    async def disconnect(self, nei: str, disconnect_msg: bool = True) -> None:
        """
        Disconnect from a neighbor.

        Args:
            nei: The neighbor to disconnect from.
            disconnect_msg: The disconnect message flag.

        """
        pass

    @abstractmethod
    def get_neighbors(self, only_direct: bool = False) -> Dict[str, Any]:
        """
        Get the neighbors.

        Args:
            only_direct: The only direct flag.

        """
        pass

    def get_address(self) -> str:
        """
        Get the address.

        Returns:
            The address.

        """
        return self.address

    @abstractmethod
    def wait_for_termination(self) -> None:
        """Wait for termination."""
        pass
