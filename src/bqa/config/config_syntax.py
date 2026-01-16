from bqa.backends import BACKEND_STR_TO_BACKEND
from bqa.config.schedule_syntax import DEFAULT_SCHEDULE, _analyse_schedule
from bqa.config.utils import (
    ConfigSyntaxError,
    _analyse_number,
    _analyse_non_neg_number,
    _analyse_non_neg_int,
    _analyse_positive_int,
    _analyse_half_to_1_number,
    _analyse_0_to_1_number,
    _get_or_default_and_warn,
    _get_or_raise,
)

# keywords

NODES_KEY = "nodes"

EDGES_KEY = "edges"

DEFAULT_FIELD_KEY = "default_field"

SCHEDULE_KEY = "schedule"

MAX_BOND_DIM_KEY = "max_bond_dim"

MAX_BP_ITER_NUMBER_KEY = "max_bp_iter_number"

SEED_KEY = "seed"

BP_EPS_KEY = "bp_eps"

PINV_EPS_KEY = "pinv_eps"

MEASUREMENT_THRESHOLD_KEY = "measurement_threshold"

DAMPING_KEY = "damping"

BACKEND_KEY = "backend" 

# defaults

DEFAULT_NODES = {}

DEFAULT_MAX_BOND_DIM = 4

DEFAULT_MAX_BP_ITERS_NUMBER = 100

DEFAULT_BP_EPS = 1e-10

DEFAULT_PINV_EPS = 1e-10

DEFAULT_BACKEND = "numpy"

DEFAULT_DEFAULT_FIELD = 0.0

DEFAULT_MEASUREMENT_THRESHOLD = 0.95

DEFAULT_SEED = 42

DEFAULT_DAMPING = 0.

def _analyse_backend(backend):
    if isinstance(backend, str):
        if backend not in BACKEND_STR_TO_BACKEND:
            raise ConfigSyntaxError(f"Unknown backend \"{backend}\", available backends {BACKEND_STR_TO_BACKEND}")
        else:
            return backend
    else:
        raise ConfigSyntaxError(f"Invalid backend \"{backend}\"")


def _analyse_nodes(nodes):
    analysed_nodes = {}

    def analyse_insert_node(node):
        if isinstance(node, (tuple, list)) and len(node) == 2:
            try:
                node_id = _analyse_non_neg_int(node[0])
                field = _analyse_number(node[1])
                if node_id in analysed_nodes:
                    raise ConfigSyntaxError(f"Duplicated node ID {node_id}")
                else:
                    analysed_nodes[node_id] = field
            except ConfigSyntaxError as e:
                raise ConfigSyntaxError(f"Invalid node {node}") from e
        else:
            raise ConfigSyntaxError(f"Invalid node {node}")

    if isinstance(nodes, (tuple, list, dict)):
        try:
            nodes_iter = nodes.items() if isinstance(nodes, dict) else nodes
            for node in nodes_iter:
                analyse_insert_node(node)
            return analysed_nodes
        except ConfigSyntaxError as e:
            raise ConfigSyntaxError(f"Invalid nodes {nodes}") from e
    else:
        raise ConfigSyntaxError(f"Invalid nodes {nodes}")


def _analyse_edge_id(edge_id):
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


def _analyse_edges(edges):
    forward_edges = {}
    backward_edges = {}

    def analyse_insert_edge(edge):
        if isinstance(edge, (tuple, list)) and len(edge) == 2:
            try:
                lhs, rhs = _analyse_edge_id(edge[0])
                coupling = _analyse_number(edge[1])
                if (
                        (lhs, rhs) in forward_edges
                        or (rhs, lhs) in forward_edges
                        or (lhs, rhs) in backward_edges
                        or (rhs, lhs) in backward_edges
                ):
                    raise ConfigSyntaxError(
                        f"Duplicated edge ID {(lhs, rhs)}, {(rhs, lhs)}"
                    )
                else:
                    forward_edges[(lhs, rhs)] = coupling
                    backward_edges[(rhs, lhs)] = coupling
            except ConfigSyntaxError as e:
                raise ConfigSyntaxError(f"Invalid edge {edge}") from e
        else:
            raise ConfigSyntaxError(f"Invalid edge {edge}")

    if isinstance(edges, (dict, tuple, list)):
        try:
            edges_iter = edges.items() if isinstance(edges, dict) else edges
            # order of edges is important, one relies on it later
            for edge in edges_iter:
                analyse_insert_edge(edge)
            return forward_edges | backward_edges
        except ConfigSyntaxError as e:
            raise ConfigSyntaxError(f"Invalid edges {edges}") from e
    else:
        raise ConfigSyntaxError(f"Invalid edges {edges}")


def _analyse_config(config):
    if isinstance(config, dict):
        try:
            nodes = _get_or_default_and_warn(config, NODES_KEY, DEFAULT_NODES)
            edges = _get_or_raise(config, EDGES_KEY)
            schedule = _get_or_default_and_warn(config, SCHEDULE_KEY, DEFAULT_SCHEDULE)
            max_bond_dim = _get_or_default_and_warn(config, MAX_BOND_DIM_KEY, DEFAULT_MAX_BOND_DIM)
            max_bp_iters_number = _get_or_default_and_warn(config, MAX_BP_ITER_NUMBER_KEY, DEFAULT_MAX_BP_ITERS_NUMBER)
            bp_eps = _get_or_default_and_warn(config, BP_EPS_KEY, DEFAULT_BP_EPS)
            pinv_eps = _get_or_default_and_warn(config, PINV_EPS_KEY, DEFAULT_PINV_EPS)
            backend = _get_or_default_and_warn(config, BACKEND_KEY, DEFAULT_BACKEND)
            default_field = _get_or_default_and_warn(config, DEFAULT_FIELD_KEY, DEFAULT_DEFAULT_FIELD)
            measurement_threshold = _get_or_default_and_warn(config, MEASUREMENT_THRESHOLD_KEY, DEFAULT_MEASUREMENT_THRESHOLD)
            seed = _get_or_default_and_warn(config, SEED_KEY, DEFAULT_SEED)
            damping = _get_or_default_and_warn(config, DAMPING_KEY, DEFAULT_DAMPING)
            return {
                NODES_KEY : _analyse_nodes(nodes),
                EDGES_KEY : _analyse_edges(edges),
                DEFAULT_FIELD_KEY : _analyse_number(default_field),
                SCHEDULE_KEY : _analyse_schedule(schedule),
                MAX_BOND_DIM_KEY : _analyse_positive_int(max_bond_dim),
                MAX_BP_ITER_NUMBER_KEY : _analyse_non_neg_int(max_bp_iters_number),
                SEED_KEY : _analyse_non_neg_int(seed),
                BP_EPS_KEY : _analyse_non_neg_number(bp_eps),
                PINV_EPS_KEY : _analyse_0_to_1_number(pinv_eps),
                MEASUREMENT_THRESHOLD_KEY : _analyse_half_to_1_number(
                    measurement_threshold
                ),
                DAMPING_KEY : _analyse_0_to_1_number(damping),
                BACKEND_KEY : _analyse_backend(backend),
            }
        except ConfigSyntaxError as e:
            raise ConfigSyntaxError(f"Invalid config {config}") from e
    else:
        raise ConfigSyntaxError(f"Config must be a dict, but got {config} of type {type(config)}")

