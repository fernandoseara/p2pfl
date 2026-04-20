#
# This file is part of the p2pfl distribution
# (see https://github.com/pguijas/p2pfl).
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
"""Launch from YAMLs."""

import importlib
import logging
import os
import time
import uuid
from typing import Any

import yaml

from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger
from p2pfl.node import Node
from p2pfl.settings import Settings
from p2pfl.utils.topologies import TopologyFactory
from p2pfl.utils.utils import wait_convergence, wait_to_finish


def load_by_package_and_name(package_name, class_name) -> Any:
    """
    Load a class by package and name.

    Args:
        package_name: The package name.
        class_name: The class name.

    """
    module = importlib.import_module(package_name)
    return getattr(module, class_name)


async def run_from_yaml(yaml_path: str, debug: bool = False) -> None:
    """
    Run a simulation from a YAML file.

    Args:
        yaml_path: The path to the YAML file.
        debug: If True, enable debug mode.

    """
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # Parse YAML configuration
    with open(yaml_path) as file:
        config = yaml.safe_load(file)

    # Update settings
    custom_settings = config.get("settings", {})
    if custom_settings:
        Settings.set_from_dict(custom_settings)
        # Refresh (already initialized)
        logger.set_level(Settings.general.LOG_LEVEL)

    # Get Amount of Nodes
    network_config = config.get("network", {})
    if not network_config:
        raise ValueError("Missing 'network' configuration in YAML file.")
    n = network_config.get("nodes")
    if not n:
        # For hierarchical topology, derive node count from clusters (+1 for root)
        hierarchy = network_config.get("hierarchy", {})
        clusters = hierarchy.get("clusters", [])
        if clusters:
            n = 1 + sum(1 + c.get("workers", 1) for c in clusters)
        else:
            raise ValueError("Missing 'nodes' under 'network' configuration in YAML file.")

    #############
    # Profiling #
    #############

    profiling = config.get("profiling", {})
    profiling_enabled = profiling.get("enabled", False)
    profiling_output_dir = profiling.get("output_dir", "profile")
    if profiling_enabled:
        import yappi  # type: ignore

        # Start profiler
        yappi.start()

    start_time = None
    if profiling.get("measure_time", False):
        start_time = time.time()

    ###################
    # Remote Loggers  #
    ###################

    remote_loggers = config.get("remote_loggers", {})
    if remote_loggers:
        logger.connect(**remote_loggers)

    ###########
    # Dataset #
    ###########

    experiment_config = config.get("experiment", {})
    dataset_config = experiment_config.get("dataset", {})  # Get dataset config
    if not dataset_config:
        raise ValueError("Missing 'dataset' configuration in YAML file.")
    data_source = dataset_config.get("source")
    if not data_source:
        raise ValueError("Missing 'source' under 'dataset' configuration in YAML file.")
    dataset_name = dataset_config.get("name")
    if not dataset_name:
        raise ValueError("Dataset source is 'huggingface' but 'name' is missing in YAML.")

    # Load data
    dataset = None
    if data_source == "huggingface":
        dataset = P2PFLDataset.from_huggingface(dataset_name)
    elif data_source == "csv":
        dataset = P2PFLDataset.from_csv(dataset_name)
    elif data_source == "json":
        dataset = P2PFLDataset.from_json(dataset_name)
    elif data_source == "parquet":
        dataset = P2PFLDataset.from_parquet(dataset_name)
    elif data_source == "pandas":
        dataset = P2PFLDataset.from_pandas(dataset_name)
    elif data_source == "custom":
        # Get custom dataset configuration
        package = dataset_config.get("package")
        dataset_class = dataset_config.get("class")
        if not package or not dataset_class:
            raise ValueError("Missing package or class for custom dataset")

        # Load custom dataset class
        dataset_class = load_by_package_and_name(package, dataset_class)
        dataset = dataset_class(**dataset_config.get("params", {}))

    if not dataset:
        print("P2PFLDataset loading process completed without creating a dataset object (check for errors above).")
        return None

    # Batch size
    dataset.set_batch_size(dataset_config.get("batch_size", 1))

    # Partitioning (do this BEFORE applying transforms)
    partitioning_config = dataset_config.get("partitioning", {})
    if not partitioning_config:
        raise ValueError("Missing 'partitioning' configuration in YAML file.")
    partition_package = partitioning_config.get("package")
    partition_class_name = partitioning_config.get("strategy")
    if not partition_package or not partition_class_name:
        raise ValueError("Missing 'partition_strategy' configuration in YAML file.")
    reduced_dataset = partitioning_config.get("reduced_dataset", False)
    reduction_factor = partitioning_config.get("reduction_factor", 1)
    partitions = dataset.generate_partitions(
        n * reduction_factor if reduced_dataset else n,
        load_by_package_and_name(
            partition_package,
            partition_class_name,
        ),
        **partitioning_config.get("params", {}),
    )

    # Transforms (apply AFTER partitioning)
    transforms_config = dataset_config.get("transforms", None)
    if transforms_config:
        transforms_package = transforms_config.get("package")
        transform_function = transforms_config.get("function")
        if not transforms_package or not transform_function:
            raise ValueError("Missing 'transforms' configuration in YAML file.")
        transform_class = load_by_package_and_name(
            transforms_package,
            transform_function,
        )
        # Apply transforms to each partition
        for partition in partitions:
            partition.set_transforms(transform_class(**transforms_config.get("params", {})))

    #########
    # Model #
    #########

    model_config = experiment_config.get("model", {})
    if not model_config:
        raise ValueError("Missing 'model' configuration in YAML file.")
    model_package = model_config.get("package")
    model_build_fn = model_config.get("model_build_fn")
    if not model_package or not model_build_fn:
        raise ValueError("Missing 'model' configuration in YAML file.")
    model_class = load_by_package_and_name(
        model_package,
        model_build_fn,
    )

    def model_fn() -> P2PFLModel:
        params = model_config.get("params", {})
        params = {**params, "compression": model_config.get("compression", None)}
        return model_class(**params)

    ##############
    # Aggregator #
    ##############

    aggregator = experiment_config.get("aggregator")
    if not aggregator:
        raise ValueError("Missing 'aggregator' configuration in YAML file.")
    aggregator_package = aggregator.get("package")
    aggregator_class_name = aggregator.get("aggregator")
    if not aggregator_package or not aggregator_class_name:
        raise ValueError("Missing 'aggregator' configuration in YAML file.")
    aggregator_class = load_by_package_and_name(
        aggregator_package,
        aggregator_class_name,
    )

    def aggregator_fn() -> Aggregator:
        return aggregator_class(**aggregator.get("params", {}))

    ############
    # Workflow #
    ############
    workflow = experiment_config.get("workflow")
    if not workflow:
        raise ValueError("Missing 'workflow' configuration in YAML file.")
    workflow_name: str = workflow

    ###########
    # Network #
    ###########

    # Load protocol
    protocol_package = network_config.get("package")
    protocol_class_name = network_config.get("protocol")
    if not protocol_package or not protocol_class_name:
        raise ValueError("Missing 'protocol' configuration in YAML file.")
    protocol = load_by_package_and_name(
        protocol_package,
        protocol_class_name,
    )

    topology = network_config.get("topology")
    if not topology:
        raise ValueError("Missing 'topology' configuration in YAML file.")

    # Create nodes
    nodes: list[Node] = []
    for i in range(n):
        node = Node(
            model_fn(),
            partitions[i],
            protocol=protocol(),
            aggregator=aggregator_fn(),
        )
        await node.start()
        nodes.append(node)

    try:
        # Start Learning
        r = experiment_config.get("rounds")
        e = experiment_config.get("epochs")
        if r < 1:
            raise ValueError("Skipping training, amount of round is less than 1")

        if topology == "hierarchical":
            await _run_hierarchical(nodes, network_config, experiment_config, workflow_name, r, e)
        else:
            await _run_flat(nodes, network_config, experiment_config, workflow_name, r, e, n)

        # Wait and check
        wait_timeout = experiment_config.get("wait_timeout", 60)
        await wait_to_finish(nodes, timeout=wait_timeout * 60, debug=debug)

    except Exception as e:
        raise e
    finally:
        # Stop Nodes
        for node in nodes:
            await node.stop()
        # Profiling
        if start_time:
            print(f"Execution time: {time.time() - start_time} seconds")
        if profiling_enabled:
            # Stop profiler
            yappi.stop()
            # Save stats
            profile_dir = os.path.join(profiling_output_dir, str(uuid.uuid4()))
            os.makedirs(profile_dir, exist_ok=True)
            for thread in yappi.get_thread_stats():
                yappi.get_func_stats(ctx_id=thread.id).save(f"{profile_dir}/{thread.name}-{thread.id}.pstat", type="pstat")
            # Print where the stats were saved
            print(f"Profile stats saved in {profile_dir}")


async def _run_flat(
    nodes: list[Node],
    network_config: dict[str, Any],
    experiment_config: dict[str, Any],
    workflow_name: str,
    r: int,
    e: int,
    n: int,
) -> None:
    """Run a flat (non-hierarchical) topology."""
    import asyncio

    topology = network_config.get("topology")
    if n > Settings.gossip.TTL:
        print(
            f"TTL less than the number of nodes ({Settings.gossip.TTL} < {n}). "
            "Some messages will not be delivered depending on the topology."
        )
    adjacency_matrix = TopologyFactory.generate_matrix(topology, len(nodes))
    await TopologyFactory.connect_nodes(adjacency_matrix, nodes)
    await wait_convergence(nodes, n - 1, only_direct=False, wait=60, debug=False)

    # Additional connections
    additional_connections = network_config.get("additional_connections")
    if additional_connections:
        for source, connect_to in additional_connections:
            await nodes[source].connect(nodes[connect_to].address)

    trainset_size = experiment_config.get("trainset_size")
    await nodes[0].set_start_learning(rounds=r, epochs=e, trainset_size=trainset_size, workflow=workflow_name)


async def _run_hierarchical(
    nodes: list[Node],
    network_config: dict[str, Any],
    experiment_config: dict[str, Any],
    workflow_name: str,
    r: int,
    e: int,
) -> None:
    """
    Run a hierarchical topology.

    Nodes are organized as: 1 root + clusters (each with 1 edge + N workers).
    The first node is the root, followed by edge/worker groups.
    Edges connect to their workers and to the root.

    Unlike flat topologies, HFL does NOT gossip the start-learning command
    because each node requires different role-specific parameters.
    """
    import asyncio

    from p2pfl.workflow.engine.experiment import Experiment

    hierarchy = network_config.get("hierarchy", {})
    clusters = hierarchy.get("clusters", [])
    edge_trains = hierarchy.get("edge_trains", True)

    if not clusters:
        raise ValueError("Missing 'clusters' in hierarchy configuration.")

    # First node is the root
    root_node = nodes[0]

    # Split remaining nodes into clusters: [edge0, w0_0, w0_1, ..., edge1, w1_0, w1_1, ...]
    edge_nodes: list[Node] = []
    worker_groups: list[list[Node]] = []
    idx = 1  # skip root
    for cluster in clusters:
        num_workers = cluster.get("workers", 1)
        edge = nodes[idx]
        workers = nodes[idx + 1 : idx + 1 + num_workers]
        edge_nodes.append(edge)
        worker_groups.append(workers)
        idx += 1 + num_workers

    all_edge_addrs = [e.address for e in edge_nodes]

    # Connect workers to their edge (bidirectional)
    for edge, workers in zip(edge_nodes, worker_groups, strict=True):
        for worker in workers:
            await worker.connect(edge.address)
            await edge.connect(worker.address)

    # Connect edges to root (bidirectional)
    for edge in edge_nodes:
        await edge.connect(root_node.address)
        await root_node.connect(edge.address)

    # Brief wait for connections to stabilize
    await asyncio.sleep(1)

    # Start learning on each node directly (no gossip) with role-specific params.
    exp_name = f"hfl-{time.time()}"

    # Start root
    root_exp = Experiment.create(
        exp_name=exp_name,
        total_rounds=r,
        epochs_per_round=e,
        workflow=workflow_name,
        is_initiator=True,
        role="root",
        child_edge_addrs=all_edge_addrs,
    )
    await root_node._start_learning_workflow(workflow_name, root_exp)

    for edge, workers in zip(edge_nodes, worker_groups, strict=True):
        worker_addrs = [w.address for w in workers]

        # Start edge
        edge_exp = Experiment.create(
            exp_name=exp_name,
            total_rounds=r,
            epochs_per_round=e,
            workflow=workflow_name,
            is_initiator=True,
            role="edge",
            worker_addrs=worker_addrs,
            root_addr=root_node.address,
            edge_trains=edge_trains,
        )
        await edge._start_learning_workflow(workflow_name, edge_exp)

        # Start workers
        for worker in workers:
            worker_exp = Experiment.create(
                exp_name=exp_name,
                total_rounds=r,
                epochs_per_round=e,
                workflow=workflow_name,
                is_initiator=False,
                role="worker",
                edge_addr=edge.address,
            )
            await worker._start_learning_workflow(workflow_name, worker_exp)
