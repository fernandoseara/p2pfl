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

"""Protobuff server."""

import asyncio
import traceback
from abc import ABC, abstractmethod
from typing import Optional, Union

import google.protobuf.empty_pb2
import grpc

from p2pfl.communication.commands.command import Command
from p2pfl.communication.protocols.protobuff.gossiper import Gossiper
from p2pfl.communication.protocols.protobuff.neighbors import Neighbors
from p2pfl.communication.protocols.protobuff.proto import node_pb2, node_pb2_grpc
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.utils.node_component import NodeComponent, allow_no_addr_check


class ProtobuffServer(ABC, node_pb2_grpc.NodeServicesServicer, NodeComponent):
    """
    Implementation of the server side logic of PROTOBUFF communication protocol.

    Args:
        addr: Address of the server.
        gossiper: Gossiper instance.
        neighbors: Neighbors instance.
        commands: List of commands to be executed by the server.

    """

    def __init__(
        self,
        neighbors: Neighbors,
        commands: Optional[list[Command]] = None,
    ) -> None:
        """Initialize the GRPC server."""
        # Message handlers
        if commands is None:
            commands = []
        self.__commands = {c.get_name(): c for c in commands}

        # (addr) Super
        NodeComponent.__init__(self)

        # Neighbors
        self._neighbors = neighbors

        # Background tasks
        self._background_tasks = set()

    ####
    # Management
    ####

    @abstractmethod
    async def start(self, wait: bool = False) -> None:
        """
        Start the server.

        Args:
            wait: If True, wait for termination.

        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the server."""
        pass

    @abstractmethod
    async def wait_for_termination(self) -> None:
        """Wait for termination."""
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """
        Check if the server is running.

        Returns:
            True if the server is running, False otherwise.

        """
        pass

    ####
    # Service Implementation (server logic on protobuff)
    ####

    async def handshake(self, request: node_pb2.HandShakeRequest, _: grpc.aio.ServicerContext) -> node_pb2.ResponseMessage:
        """
        Service. It is called when a node connects to another.

        Args:
            request: Request message.
            _: Context.

        """
        if await self._neighbors.add(request.addr, non_direct=False, handshake=False):
            return node_pb2.ResponseMessage()
        else:
            return node_pb2.ResponseMessage(error="Cannot add the node (duplicated or wrong direction)")

    async def disconnect(self, request: node_pb2.HandShakeRequest, _: grpc.aio.ServicerContext) -> google.protobuf.empty_pb2.Empty:
        """
        Service. It is called when a node disconnects from another.

        Args:
            request: Request message.
            _: Context.

        """
        await self._neighbors.remove(request.addr, disconnect_msg=False)
        return google.protobuf.empty_pb2.Empty()

    async def send(self, request: node_pb2.RootMessage, _: grpc.aio.ServicerContext) -> node_pb2.ResponseMessage:
        """
        Service. Handles both regular messages and model weights.

        Args:
            request: The RootMessage containing either a Message or Weights payload.
            _: Context.

        """
        # If message already processed, return
        # if request.HasField("message"):
        #     """
        #     if request.cmd != "beat" or (not Settings.heartbeat.EXCLUDE_BEAT_LOGS and request.source == "beat"):
        #         logger.debug(self.addr, f"🙅 Message already processed: {request.cmd} (id {request.message.hash})")
        #     """
        #     return node_pb2.ResponseMessage()

        # Process message/model
        if request.cmd != "beat" or (not Settings.heartbeat.EXCLUDE_BEAT_LOGS and request.cmd == "beat"):
            emoji = "📫" if request.HasField("message") else "📦"
            logger.debug(
                self.addr,
                f"{emoji} {request.cmd.upper()} received from {request.source}",
            )
        if request.cmd in self.__commands:
            try:
                if request.HasField("message"):
                    task = asyncio.create_task(self.__commands[request.cmd].execute(request.source, request.round, *request.message.args))
                elif request.HasField("weights"):
                    task = asyncio.create_task(self.__commands[request.cmd].execute(
                        request.source,
                        request.round,
                        weights=request.weights.weights,
                        contributors=request.weights.contributors,
                        num_samples=request.weights.num_samples,
                    ))
                else:
                    error_text = f"Error while processing command: {request.cmd}: No message or weights"
                    logger.error(self.addr, error_text)
                    return node_pb2.ResponseMessage(error=error_text)

                # Add task to the set. This creates a strong reference.
                self._background_tasks.add(task)

                # To prevent keeping references to finished tasks forever,
                # make each task remove its own reference from the set after
                # completion:
                task.add_done_callback(self._background_tasks.discard)
            except Exception as e:
                error_text = f"Error while processing command: {request.cmd}. {type(e).__name__}: {e}"
                logger.error(self.addr, error_text + f"\n{traceback.format_exc()}")
                return node_pb2.ResponseMessage(error=error_text)
        else:
            # disconnect node
            logger.error(self.addr, f"Unknown command: {request.cmd} from {request.source}")
            return node_pb2.ResponseMessage(error=f"Unknown command: {request.cmd}")

        # If message gossip
        if request.HasField("message") and request.message.ttl > 0:
            # Update ttl and gossip
            request.message.ttl -= 1
            for addr, _ in self._neighbors.get_all(only_direct=True).items():
                # Send message to all direct neighbors
                if addr != request.source and addr != self.addr:
                    try:
                        await self._neighbors.get(addr).send(request)
                    except Exception as e:
                        logger.error(self.addr, f"Error while sending message to {addr}: {e}")

        return node_pb2.ResponseMessage()

    ####
    # Commands
    ####

    @allow_no_addr_check
    def add_command(self, cmds: Union[Command, list[Command]]) -> None:
        """
        Add a command.

        Args:
            cmds: Command or list of commands to be added.

        """
        if isinstance(cmds, list):
            for cmd in cmds:
                self.__commands[cmd.get_name()] = cmd
        elif isinstance(cmds, Command):
            self.__commands[cmds.get_name()] = cmds
        else:
            raise Exception("Command not valid")
