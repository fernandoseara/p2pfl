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

"""Component of a node (Learner, Aggregator, Communication Protocol...)."""

from typing import Any, Callable


def allow_no_addr_check(method: Callable[..., Any]) -> Callable[..., Any]:
    """Decorate to mark a method as exempt from the addr check."""
    method.__no_addr_check__ = True  # type: ignore
    return method

class NodeComponent:
    """
    Component of a node (Learner, Aggregator, Communication Protocol...).

    Attributes:
        addr: The address of the node (must be a non-empty string).

    """

    address: str

    def set_addr(self, address: str) -> str:
        """Set the address of the node."""
        self.address = address
        return self.address
