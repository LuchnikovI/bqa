from functools import reduce
from typing import Any
from bqa.config.schedule_syntax import DEFAULT_SCHEDULE, _analyse_schedule
from bqa.config.utils import (
    ConfigSyntaxError,
    _analyse_number,
    _analyse_non_neg_number,
    _analyse_non_neg_int,
    _analyse_positive_int,
    _analyse_half_to_1_number,
    _ok_or_default_and_warn,
    _unwrap_or,
)

# types

Edge = tuple[int, int]

NodeToAmpl = dict[int, float]

EdgeToAmpl = dict[Edge, float]

Config = dict[str, Any]

# defaults

DEFAULT_NODES = []

DEFAULT_MAX_BOND_DIM = 4

DEFAULT_MAX_BP_ITERS_NUMBER = 100

DEFAULT_BP_EPS = 1e-5

DEFAULT_BACKEND = "numpy"

DEFAULT_DEFAULT_FIELD = 0.0

DEFAULT_MEASUREMENT_THRESHOLD = 0.95

DEFAULT_SEED = 42

# syntax analysis

BACKENDS = {"numpy"}

def _analyse_backend(backend) -> str:
    if isinstance(backend, str):
        if backend not in BACKENDS:
            raise ConfigSyntaxError(f"Unknown backend {backend}")
        else:
            return backend
    else:
        raise ConfigSyntaxError(f"Invalid backend {backend}")


def _add_analyse_node(nodes: NodeToAmpl, node) -> NodeToAmpl:
    if isinstance(node, (tuple, list)) and len(node) == 2:
        try:
            node_id = _analyse_non_neg_int(node[0])
            field = _analyse_number(node[1])
            if node_id in nodes:
                raise ConfigSyntaxError(f"Duplicated node ID {node_id}")
            else:
                nodes[node_id] = field
                return nodes
        except ConfigSyntaxError as e:
            raise ConfigSyntaxError(f"Invalid node {node}") from e
    else:
        raise ConfigSyntaxError(f"Invalid node {node}")


def _analyse_nodes(nodes) -> NodeToAmpl:
    if isinstance(nodes, (tuple, list, dict)):
        try:
            nodes_iter = nodes.items() if isinstance(nodes, dict) else nodes
            return reduce(_add_analyse_node, nodes_iter, {})
        except ConfigSyntaxError as e:
            raise ConfigSyntaxError(f"Invalid nodes {nodes}") from e
    else:
        raise ConfigSyntaxError(f"Invalid nodes {nodes}")


def _analyse_edge_id(edge_id) -> tuple[int, int]:
    if isinstance(edge_id, (list, tuple)) and len(edge_id) == 2:
        try:
            lhs = _analyse_non_neg_int(edge_id[0])
            rhs = _analyse_non_neg_int(edge_id[1])
            if lhs == rhs:
                raise ConfigSyntaxError(
                    f"LHS and RHS of the edge ID must not be equal, got edge ID {edge_id}"
                )
            return (lhs, rhs)
        except ConfigSyntaxError as e:
            raise ConfigSyntaxError(f"Invalid edge ID {edge_id}") from e
    else:
        raise ConfigSyntaxError(f"Invalid edge ID {edge_id}")


def _add_analyse_edge(edges: tuple[EdgeToAmpl, EdgeToAmpl], edge) -> tuple[EdgeToAmpl, EdgeToAmpl]:
    forward_edges, backward_edges = edges
    if isinstance(edge, (tuple, list)) and len(edge) == 2:
        try:
            lhs, rhs = _analyse_edge_id(edge[0])
            coupling = _analyse_number(edge[1])
            if (lhs, rhs) in forward_edges or (rhs, lhs) in forward_edges \
               or (lhs, rhs) in backward_edges or (rhs, lhs) in backward_edges:
                raise ConfigSyntaxError(
                    f"Duplicated edge ID {(lhs, rhs)}, {(rhs, lhs)}"
                )
            else:
                forward_edges[(lhs, rhs)] = coupling
                backward_edges[(rhs, lhs)] = coupling
                return forward_edges, backward_edges
        except ConfigSyntaxError as e:
            raise ConfigSyntaxError(f"Invalid edge {edge}") from e
    else:
        raise ConfigSyntaxError(f"Invalid edge {edge}")


def _analyse_edges(edges) -> EdgeToAmpl:
    if isinstance(edges, (dict, tuple, list)):
        try:
            edges_iter = edges.items() if isinstance(edges, dict) else edges
            # order of edges is important, one relies on it
            forward_edges, backward_edges = reduce(_add_analyse_edge, edges_iter, ({}, {}))
            return forward_edges | backward_edges
        except ConfigSyntaxError as e:
            raise ConfigSyntaxError(f"Invalid edges {edges}") from e
    else:
        raise ConfigSyntaxError(f"Invalid edges {edges}")


def _get_field_or_default_and_warn(config: dict, field: str, default, msg: str):
    return _ok_or_default_and_warn(config.get(field), default, msg)


def _analyse_config(config) -> Config:
    if isinstance(config, dict):
        try:
            nodes = _get_field_or_default_and_warn(
                config,
                "nodes",
                DEFAULT_NODES,
                f"`nodes` field is missing in config, set to {DEFAULT_NODES}",
            )
            edges = _unwrap_or(config.get("edges"), "`edges` field is missing")
            schedule = _get_field_or_default_and_warn(
                config,
                "schedule",
                DEFAULT_SCHEDULE,
                f"`schedule` field is missing, set to {DEFAULT_SCHEDULE}",
            )
            max_bond_dim = _get_field_or_default_and_warn(
                config,
                "max_bond_dim",
                DEFAULT_MAX_BOND_DIM,
                f"`max_bond_dim` field is missing, set to {DEFAULT_MAX_BOND_DIM}",
            )
            max_bp_iters_number = _get_field_or_default_and_warn(
                config,
                "max_bp_iters_number",
                DEFAULT_MAX_BP_ITERS_NUMBER,
                f"`max_bp_iters_number` field is missing, set to {DEFAULT_MAX_BP_ITERS_NUMBER}",
            )
            bp_eps = _get_field_or_default_and_warn(
                config,
                "bp_eps",
                DEFAULT_BP_EPS,
                f"`bp_eps` field is missing, set to {DEFAULT_BP_EPS}",
            )
            backend = _get_field_or_default_and_warn(
                config,
                "backend",
                DEFAULT_BACKEND,
                f"`backend` field is missing, set to {DEFAULT_BACKEND}",
            )
            default_field = _get_field_or_default_and_warn(
                config,
                "default_field",
                DEFAULT_DEFAULT_FIELD,
                f"`default_field` field is missing, set to  {DEFAULT_DEFAULT_FIELD}",
            )
            measurement_threshold = _get_field_or_default_and_warn(
                config,
                "measurement_threshold",
                DEFAULT_MEASUREMENT_THRESHOLD,
                f"`measurement_threshold` field is missing, set to {DEFAULT_MEASUREMENT_THRESHOLD}"
            )
            seed = _get_field_or_default_and_warn(
                config,
                "seed",
                DEFAULT_SEED,
                f"`seed` field is missing, set to {DEFAULT_SEED}"
            )
            return {
                "nodes": _analyse_nodes(nodes),
                "edges": _analyse_edges(edges),
                "default_field": _analyse_number(default_field),
                "schedule": _analyse_schedule(schedule),
                "max_bond_dim": _analyse_positive_int(max_bond_dim),
                "max_bp_iters_number": _analyse_non_neg_int(max_bp_iters_number),
                "seed" : _analyse_non_neg_int(seed),
                "bp_eps": _analyse_non_neg_number(bp_eps),
                "measurement_threshold" : _analyse_half_to_1_number(measurement_threshold),
                "backend": _analyse_backend(backend),
            }
        except ConfigSyntaxError as e:
            raise ConfigSyntaxError(f"Invalid config {config}") from e
    else:
        raise ConfigSyntaxError(f"Invalid config {config}")
