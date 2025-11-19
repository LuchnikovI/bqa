from functools import reduce
from typing import Any
from bqa.config_syntax.schedule_syntax import DEFAULT_SCHEDULE, analyse_schedule
from bqa.config_syntax.utils import (ConfigSyntaxError, analyse_float, analyse_non_neg_float,
                                     analyse_non_neg_int, analyse_positive_int, get_default_and_warn,
                                     unwrap_or)

# types

Node = tuple[int, float]

Nodes = dict[int, float]

Edge = tuple[tuple[int, int], float]

Edges = dict[tuple[int, int], float]

Config = dict[str, Any]

# defaults

DEFAULT_NODES = []

DEFAULT_MAX_BOND_DIM = 4

DEFAULT_MAX_BP_ITERS_NUMBER = 100

DEFAULT_BP_EPS = 1e-5

DEFAULT_BACKEND = "numpy"

DEFAULT_DEFAULT_FIELD = 0.

# syntax analysis

BACKENDS = {"numpy", "cupy"}

def analyse_backend(backend) -> str:
    if isinstance(backend, str):
        if backend not in BACKENDS:
            raise ConfigSyntaxError(f"Unknown backend {backend}")
        else:
            return backend
    else:
        raise ConfigSyntaxError(f"Invalid backend {backend}")

def add_analyse_node(nodes: Nodes, node) -> Nodes:
    if isinstance(node, (tuple, list)) and len(node) == 2:
        try:
            node_id = analyse_non_neg_int(node[0])
            field = analyse_float(node[1])
            if node_id in nodes:
                raise ConfigSyntaxError(f"Duplicated node ID {node_id}")
            else:
                nodes[node_id] = field
                return nodes
        except ConfigSyntaxError as e:
            raise ConfigSyntaxError(f"Invalid node {node}") from e
    else:
        raise ConfigSyntaxError(f"Invalid node {node}")

def analyse_nodes(nodes) -> Nodes:
    if isinstance(nodes, (tuple, list, dict)):
        try:
            nodes_iter = nodes.items() if isinstance(nodes, dict) else nodes
            return reduce(add_analyse_node, nodes_iter, {})
        except ConfigSyntaxError as e:
            raise ConfigSyntaxError(f"Invalid nodes {nodes}") from e
    else:
        raise ConfigSyntaxError(f"Invalid nodes {nodes}")

def analyse_edge_id(edge_id) -> tuple[int, int]:
    if isinstance(edge_id, (list, tuple)) and len(edge_id) == 2:
        try:
            lhs = analyse_non_neg_int(edge_id[0])
            rhs = analyse_non_neg_int(edge_id[1])
            if lhs == rhs:
                raise ConfigSyntaxError(
                    f"LHS and RHS of the edge ID must not be equal, got edge ID {edge_id}"
                )
            return (lhs, rhs)
        except ConfigSyntaxError as e:
            raise ConfigSyntaxError(f"Invalid edge ID {edge_id}") from e
    else:
        raise ConfigSyntaxError(f"Invalid edge ID {edge_id}")

def add_analyse_edge(edges: Edges, edge) -> Edges:
    if isinstance(edge, (tuple, list)) and len(edge) == 2:
        try:
            (lhs, rhs) = analyse_edge_id(edge[0])
            coupling = analyse_float(edge[1])
            if (lhs, rhs) in edges or (rhs, lhs) in edges:
                raise ConfigSyntaxError(f"Duplicated edge ID {(lhs, rhs)}, {(rhs, lhs)}")
            else:
                edges[(lhs, rhs)] = coupling
                edges[(rhs, lhs)] = coupling
                return edges
        except ConfigSyntaxError as e:
            raise ConfigSyntaxError(f"Invalid edge {edge}") from e
    else:
        raise ConfigSyntaxError(f"Invalid edge {edge}")

def analyse_edges(edges) -> Edges:
    if isinstance(edges, (dict, tuple, list)):
        try:
            edges_iter = edges.items() if isinstance(edges, dict) else edges
            return reduce(add_analyse_edge, edges_iter, {})
        except ConfigSyntaxError as e:
            raise ConfigSyntaxError(f"Invalid edges {edges}") from e
    else:
        raise ConfigSyntaxError(f"Invalid edges {edges}")

def analyse_config(config) -> Config:
    if isinstance(config, dict):
        try:
            nodes = config.get("nodes") \
                or get_default_and_warn(DEFAULT_NODES, \
                f"`nodes` field is missing in config, set to {DEFAULT_NODES}")
            edges = unwrap_or(config.get("edges"), "`edges` field is missing")
            schedule = config.get("schedule") \
                or get_default_and_warn(DEFAULT_SCHEDULE, \
                f"`schedule` field is missing, set to {DEFAULT_SCHEDULE}")
            max_bond_dim = config.get("max_bond_dim") \
                or get_default_and_warn(DEFAULT_MAX_BOND_DIM, \
                f"`max_bond_dim` filed is missing, set to {DEFAULT_MAX_BOND_DIM}")
            max_bp_iters_number = config.get("max_bp_iters_number") \
                or get_default_and_warn(DEFAULT_MAX_BP_ITERS_NUMBER , \
                f"`max_bp_iters_number` field is missing, set to {DEFAULT_MAX_BP_ITERS_NUMBER}")
            bp_eps = config.get("bp_eps") \
                or get_default_and_warn(DEFAULT_BP_EPS, \
                f"`bp_eps` field is missing, set to {DEFAULT_BP_EPS}")
            backend = config.get("backend") \
                or get_default_and_warn(DEFAULT_BACKEND, \
                f"`backend` field is missing, set to {DEFAULT_BACKEND}")
            default_field = config.get("default_field") \
                or get_default_and_warn(DEFAULT_DEFAULT_FIELD, \
                f"`default_field` field is missing, set to  {DEFAULT_DEFAULT_FIELD}")
            return {"nodes" : analyse_nodes(nodes),
                    "edges" : analyse_edges(edges),
                    "default_field" : analyse_float(default_field),
                    "schedule" : analyse_schedule(schedule),
                    "max_bond_dim" : analyse_positive_int(max_bond_dim),
                    "max_bp_iters_number" : analyse_non_neg_int(max_bp_iters_number),
                    "bp_eps" : analyse_non_neg_float(bp_eps),
                    "backend" : analyse_backend(backend)}
        except ConfigSyntaxError as e:
            raise ConfigSyntaxError(f"Invalid config {config}") from e
    else:
        raise ConfigSyntaxError(f"Invalid config {config}")

