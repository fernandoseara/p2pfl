#
# This file is part of the p2pfl (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2026 Pedro Guijas Bravo.
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
"""Typed workflow context base class."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
    from p2pfl.learning.aggregators.aggregator import Aggregator
    from p2pfl.learning.frameworks.learner import Learner

from p2pfl.workflow.engine.experiment import Experiment


@dataclass
class WorkflowContext:
    """
    Base typed context injected into all workflow stages.

    This is **not** meant to pass arguments from the node to the workflow.
    For hyperparameters or initial arguments, use ``Experiment`` (via
    ``experiment.data``).  The context only holds shared components that
    stages need at runtime (learner, aggregator, communication, etc.).

    Subclass this to add workflow-specific **mutable state** that stages
    share during execution (e.g. peers, train_set).

    Example::

        @dataclass
        class MyContext(WorkflowContext):
            peers: dict[str, PeerState] = field(default_factory=dict)
            custom_flag: bool = False
    """

    address: str
    learner: Learner
    aggregator: Aggregator
    cp: CommunicationProtocol
    generator: random.Random
    experiment: Experiment


TContext = TypeVar("TContext", bound=WorkflowContext)
