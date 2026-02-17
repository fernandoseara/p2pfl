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

import asyncio
from abc import ABCMeta
from collections.abc import Callable
from functools import wraps
from typing import Any


def allow_no_addr_check(method: Callable[..., Any]) -> Callable[..., Any]:
    """Decorate to mark a method as exempt from the address check."""
    method.__no_addr_check__ = True  # type: ignore
    return method


class AddressRequiredMeta(ABCMeta):
    """Metaclass to ensure that the address is set before any method is called, unless the method is marked with @allow_no_addr_check."""

    def __new__(cls, name: str, bases: tuple[type, ...], dct: dict[str, Any]) -> Any:
        """Create a new class with methods wrapped to ensure the address is set."""
        for attr_name, attr_value in dct.items():
            # Skip staticmethod and classmethod - they don't have 'self'
            if isinstance(attr_value, staticmethod | classmethod):
                continue
            if callable(attr_value) and attr_name != "set_address" and attr_name != "__init__":
                dct[attr_name] = cls.ensure_address_set(attr_value)
        return super().__new__(cls, name, bases, dct)

    @staticmethod
    def ensure_address_set(method: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a method to ensure the address is set before it is called, unless the method is decorated with @allow_no_addr_check."""
        # Check if the method is async and create appropriate wrapper
        if asyncio.iscoroutinefunction(method):

            @wraps(method)
            async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                if hasattr(method, "__no_addr_check__"):
                    return await method(self, *args, **kwargs)
                if not hasattr(self, "address") or self.address == "":
                    raise ValueError("Address must be set before calling this method.")
                return await method(self, *args, **kwargs)

            return async_wrapper
        else:

            @wraps(method)
            def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                if hasattr(method, "__no_addr_check__"):
                    return method(self, *args, **kwargs)
                if not hasattr(self, "address") or self.address == "":
                    raise ValueError("Address must be set before calling this method.")
                return method(self, *args, **kwargs)

            return wrapper

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        """Create an instance of the class and initialize the address attribute if not already set."""
        instance = super().__call__(*args, **kwargs)
        if not hasattr(instance, "address") or not instance.address:
            instance.address = ""
        return instance

    def set_address(cls, instance: Any, address: str) -> None:
        """Set the address of the instance."""
        instance.address = address


class NodeComponent(metaclass=AddressRequiredMeta):
    """
    Component of a node (Learner, Aggregator, Communication Protocol...).

    Attributes:
        address: The address of the node (must be a non-empty string).

    """

    address: str

    def set_address(self, address: str) -> str:
        """Set the address of the node."""
        self.address = address
        return self.address
