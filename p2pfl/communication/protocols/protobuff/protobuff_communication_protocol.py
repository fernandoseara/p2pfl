#
# This file is part of the federated_learning_p2p (p2pfl) distribution (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2022 Pedro Guijas Bravo.
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

"""GRPC communication protocol."""

import asyncio
import random
from abc import abstractmethod
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Optional, Union

from p2pfl.communication.commands.command import Command
from p2pfl.communication.commands.message.heartbeat_command import HeartbeatCommand
from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
from p2pfl.communication.protocols.exceptions import CommunicationError, ProtocolNotStartedError
from p2pfl.communication.protocols.protobuff.client import ProtobuffClient
from p2pfl.communication.protocols.protobuff.gossiper import Gossiper
from p2pfl.communication.protocols.protobuff.heartbeater import Heartbeater
from p2pfl.communication.protocols.protobuff.neighbors import Neighbors
from p2pfl.communication.protocols.protobuff.proto import node_pb2
from p2pfl.communication.protocols.protobuff.server import ProtobuffServer
from p2pfl.settings import Settings
from p2pfl.utils.node_component import allow_no_addr_check


def running(func):
    """Ensure that the server is running before executing a method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self._server.is_running():
            raise ProtocolNotStartedError("The protocol has not been started.")
        return func(self, *args, **kwargs)

    return wrapper


class ProtobuffCommunicationProtocol(CommunicationProtocol):
    """
    Protobuff communication protocol.

    Args:
        addr: Address of the node.
        commands: Commands to add to the communication protocol.

    .. todo:: https://grpc.github.io/grpc/python/grpc_asyncio.html
    .. todo:: Decouple the heeartbeat command.

    """

    def __init__(
        self,
        commands: Optional[list[Command]] = None,
    ) -> None:
        """Initialize the GRPC communication protocol."""
        # (addr) Super
        CommunicationProtocol.__init__(self)
        # Neighbors
        self._neighbors: Neighbors = Neighbors(self.build_client)
        # Gossip
        self._gossiper = Gossiper(self._neighbors)
        # GRPC
        self._server: ProtobuffServer = self.build_server(self._gossiper, self._neighbors, commands)
        # Hearbeat
        self._heartbeater: Heartbeater = Heartbeater(self._neighbors, self.build_msg)
        # Commands
        self.add_command(HeartbeatCommand(self._heartbeater))
        if commands is None:
            commands = []
        self.add_command(commands)

    @allow_no_addr_check
    @abstractmethod
    def build_client(self, *args, **kwargs) -> ProtobuffClient:
        """Build client function."""
        pass

    @allow_no_addr_check
    @abstractmethod
    def build_server(self, *args, **kwargs) -> ProtobuffServer:
        """Build server function."""
        pass

    def set_addr(self, addr: str) -> str:
        """Set the addr of the node."""
        # Delegate on server
        addr = self._server.set_addr(addr)
        # Update components
        self._neighbors.set_addr(addr)
        self._heartbeater.set_addr(addr)
        self._gossiper.set_addr(addr)
        # Set on super
        return super().set_addr(addr)

    async def start(self) -> None:
        """Start the GRPC communication protocol."""
        await self._server.start()
        await self._heartbeater.start()
        await self._gossiper.start()

    @running
    async def stop(self) -> None:
        """Stop the GRPC communication protocol."""
        # Run the stop methods of async tasks, awaiting their completion
        await self._heartbeater.stop()
        await self._gossiper.stop()

        # Clear neighbors and stop the server
        await self._neighbors.clear_neighbors()
        await self._server.stop()

    @allow_no_addr_check
    def add_command(self, cmds: Union[Command, list[Command]]) -> None:
        """
        Add a command to the communication protocol.

        Args:
            cmds: The command to add.

        """
        self._server.add_command(cmds)

    @allow_no_addr_check
    def remove_command(self, cmd: Union[str, Command]) -> None:
        """
        Remove a command from the communication protocol.

        Args:
            cmd: The command to remove.

        """
        self._server.remove_command(cmd)

    @running
    async def connect(self, addr: str, non_direct: bool = False) -> bool:
        """
        Connect to a neighbor.

        Args:
            addr: The address to connect to.
            non_direct: The non direct flag.

        """
        return await self._neighbors.add(addr, non_direct=non_direct)

    @running
    async def disconnect(self, nei: str, disconnect_msg: bool = True) -> None:
        """
        Disconnect from a neighbor.

        Args:
            nei: The neighbor to disconnect from.
            disconnect_msg: The disconnect message flag.

        """
        await self._neighbors.remove(nei, disconnect_msg=disconnect_msg)

    def _build_node_pb2_message(self,
        cmd: str,
        args: Optional[list[str]] = None,
        round: Optional[int] = None,
        ttl: int = 0
        ) -> node_pb2.Message:
        """
        Build a Message to send to the neighbors.

        Args:
            cmd: Command of the message.
            args: Arguments of the message.
            round: Round of the message.

        Returns:
            Message to send.

        """
        if round is None:
            round = -1
        if args is None:
            args = []
        hs = hash(str(cmd) + str(args) + str(datetime.now()) + str(random.randint(0, 100000)))
        args = [str(a) for a in args]

        return node_pb2.Message(
                ttl=ttl,
                hash=hs,
                args=args,
            )

    def build_msg(self, command_name: str, content: Optional[list[Any]] = None, round: Optional[int] = None) -> node_pb2.RootMessage:
        """
        Build a RootMessage to send to the neighbors.

        Args:
            command_name: Command of the message.
            content: Content of the message.
            round: Round of the message.

        Returns:
            RootMessage to send.

        """
        # Convert content to string
        if content is not None:
            content = [str(item) for item in content]

        return node_pb2.RootMessage(
            source=self.address,
            round=round,
            cmd=command_name,
            message=self._build_node_pb2_message(command_name, content, round),
        )

    def build_weights(
        self,
        cmd: str,
        round: int,
        serialized_model: bytes,
        contributors: Optional[list[str]] = None,
        weight: int = 1,
    ) -> node_pb2.RootMessage:
        """
        Build a RootMessage with a Weights payload to send to the neighbors.

        Args:
            cmd: Command of the message.
            round: Round of the message.
            serialized_model: Serialized model to send.
            contributors: List of contributors.
            weight: Weight of the message (number of samples).

        Returns:
            RootMessage to send.

        """
        if contributors is None:
            contributors = []
        return node_pb2.RootMessage(
            source=self.address,
            round=round,
            cmd=cmd,
            weights=node_pb2.Weights(
                weights=serialized_model,
                contributors=contributors,
                num_samples=weight,
            ),
        )

    @running
    async def send(
        self,
        nei: str,
        msg: Union[node_pb2.RootMessage],
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
            temporal_connection: If temporal connection.

        """
        try:
            await self._neighbors.get(nei).send(msg, temporal_connection=temporal_connection, raise_error=raise_error, disconnect_on_error=remove_on_error)
        except CommunicationError as e:
            if remove_on_error:
                await self._neighbors.remove(nei)
            if raise_error:
                raise e

    @running
    async def broadcast(self, msg: node_pb2.RootMessage, node_list: Optional[list[str]] = None) -> None:
        """
        Broadcast a message to all neighbors.

        Args:
            msg: The message to broadcast.
            node_list: Optional node list.

        """
        neis = self._neighbors.get_all(only_direct=True)
        neis_clients = [nei[0] for nei in neis.values()]

        await asyncio.gather(*(nei.send(msg) for nei in neis_clients))

    @running
    async def gossip(
        self,
        nei: str,
        msg: Union[node_pb2.RootMessage],
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
            temporal_connection: If temporal connection.

        """
        msg.message.ttl = Settings.gossip.TTL

        await self.send(nei, msg, raise_error=raise_error, remove_on_error=remove_on_error, temporal_connection=temporal_connection)

    @running
    async def broadcast_gossip(self, msg: node_pb2.RootMessage, node_list: Optional[list[str]] = None) -> None:
        """
        Gossip a message to all neighbors.

        Args:
            msg: The message to gossip.
            node_list: Optional node list.

        """
        msg.message.ttl = Settings.gossip.TTL

        neis = self._neighbors.get_all(only_direct=True)
        neis_clients = [nei[0] for nei in neis.values()]

        await asyncio.gather(*(nei.send(msg) for nei in neis_clients))


    @running
    def get_neighbors(self, only_direct: bool = False) -> dict[str, Any]:
        """
        Get the neighbors.

        Args:
            only_direct: The only direct flag.

        """
        return self._neighbors.get_all(only_direct)

    @running
    def wait_for_termination(self) -> None:
        """
        Get the neighbors.

        Args:
            only_direct: The only direct flag.

        """
        self._server.wait_for_termination()
