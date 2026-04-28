import logging
from math import isclose, isfinite
from bqa.backends import BACKEND_STR_TO_BACKEND
from bqa.config.schedule_syntax import DEFAULT_WEIGHT

class ConfigSyntaxError(ValueError):
    pass

# keys

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

FINAL_MIXING_KEY = "final_mixing"

INITIAL_MIXING_KEY = "initial_mixing"

STARTING_MIXING_KEY = "starting_mixing"

TOTAL_TIME_KEY = "total_time"

ACTIONS_KEY = "actions"

WEIGHT_KEY = "weight"

STEPS_NUMBER_KEY = "steps_number"

SPARSIFICATION_KEY = "sparsification"

CLUSTER_COUPLING_AMPLITUDE_KEY = "cluster_coupling_amplitude"

EPS_KEY = "eps"

# action types

GET_BLOCH_VECTORS = "get_bloch_vectors"

MEASURE = "measure"

SIMPLE_ACTION_TYPES = {MEASURE, GET_BLOCH_VECTORS}

log = logging.getLogger(__name__)

def validate_if_present(data, key, validator_fn):
    if key in data:
        try:
            validator_fn(data[key])
        except ConfigSyntaxError as e:
            raise ConfigSyntaxError(f"Key `{key}` contains invalid value") from e
    else:
        log.warning(f"Parameter `{key}` is not set, will be a default value")


def validate_all_records(config, key_and_validator):
    for key, validator_fn in key_and_validator:
        validate_if_present(config, key, validator_fn)


def validate_int(value):
    if not isinstance(value, int):
        raise ConfigSyntaxError(f"Must be an integer, but received value of type `{type(value)}`")


def validate_number(value):
    if not isinstance(value, (int, float)):
        raise ConfigSyntaxError(f"Must be either integer or a float, but received value of type `{type(value)}`")
    if isinstance(value, float) and not isfinite(value):
        raise ConfigSyntaxError(f"Invalid float {value}")


def validate_non_neg(value):
    if value < 0:
        raise ConfigSyntaxError(f"Must be >= 0, but received {value}")


def validate_positive(value):
    if value <= 0:
        raise ConfigSyntaxError(f"Must be > 0, but received {value}")


def validate_non_neg_int(value):
    validate_int(value)
    validate_non_neg(value)


def validate_positive_int(value):
    validate_int(value)
    validate_positive(value)


def validate_0_to_1_number(value):
    validate_number(value)
    if value < 0.0 or value > 1.0:
        raise ConfigSyntaxError(f"Must be a value from [0, 1] but received value {value}")


def validate_half_to_1_number(value):
    validate_number(value)
    if value < 0.5 or value > 1.0:
        raise ConfigSyntaxError(f"Must be a value from [0.5, 1] but received value {value}")


def validate_number_1_to_inf(value):
    validate_number(value)
    if value < 1:
        raise ConfigSyntaxError(f"Must be >= 1, but received {value}")


def validate_positive_number(value):
    validate_number(value)
    validate_positive(value)


def validate_container(value):
    if not isinstance(value, (dict, list, tuple)):
        raise ConfigSyntaxError(f"Must be either dictionary, list or tuple, but received value of type `{type(value)}`")


def validate_sequence(value):
    if not isinstance(value, (list, tuple)):
        raise ConfigSyntaxError(f"Must be either list or tuple, but received a value of type `{type(value)}`")


def validate_pair(value):
    validate_sequence(value)
    if len(value) != 2:
        raise ConfigSyntaxError(f"Must be of length 2, but received a sequence of length {len(value)}")


def validate_edge_id(edge_id):
    try:
        validate_pair(edge_id)
        lhs_id, rhs_id = edge_id
        validate_non_neg_int(lhs_id)
        validate_non_neg_int(rhs_id)
        if lhs_id == rhs_id:
            raise ConfigSyntaxError(f"Left and right node IDs must not be equal, but received edge ID is `{(lhs_id, rhs_id)}`")
    except ConfigSyntaxError as e:
        raise ConfigSyntaxError("Invalid edge ID") from e


def validate_edges(edges):
    validate_container(edges)
    seen_edges = {}
    for num, edge in enumerate(edges.items() if isinstance(edges, dict) else edges):
        try:
            validate_pair(edge)
            edge_id, cpl = edge
            validate_edge_id(edge_id)
            validate_number(cpl)
            lhs_id, rhs_id = edge_id
        except ConfigSyntaxError as e:
            raise ConfigSyntaxError(f"Invalid edge number {num}") from e
        if (lhs_id, rhs_id) in seen_edges and seen_edges[(lhs_id, rhs_id)] != cpl:
            raise ConfigSyntaxError(f"An attempt to rewrite the coupling value from {seen_edges[(lhs_id, rhs_id)]} to {cpl} for edge number {num} with ID {(lhs_id, rhs_id)}")
        if (rhs_id, lhs_id) in seen_edges and seen_edges[(rhs_id, lhs_id)] != cpl:
            raise ConfigSyntaxError(f"An attempt to rewrite the coupling value from {seen_edges[(rhs_id, lhs_id)]} to {cpl} for edge number {num} with ID {(rhs_id, lhs_id)}")
        seen_edges[(lhs_id, rhs_id)] = cpl


def validate_nodes(nodes):
    validate_container(nodes)
    seen_nodes = {}
    for num, node in enumerate(nodes.items() if isinstance(nodes, dict) else nodes):
        try:
            validate_pair(node)
            node_id, field = node
            validate_non_neg_int(node_id)
            validate_number(field)
        except ConfigSyntaxError as e:
            raise ConfigSyntaxError(f"Invalid node number {num}") from e
        if node_id in seen_nodes and seen_nodes[node_id] != field:
            raise ConfigSyntaxError(f"An attempt to rewrite the field value from {seen_nodes[node_id]} to {field} at node number {node_id}")
        seen_nodes[node_id] = field


def validate_backend(backend):
    if not isinstance(backend, str):
        raise ConfigSyntaxError(f"Must be a string, but received value of type `{type(backend)}`")
    if backend not in BACKEND_STR_TO_BACKEND:
        raise ConfigSyntaxError(f"Must be a value from {list(BACKEND_STR_TO_BACKEND.keys())}, but received {backend}")


def validate_actions(actions):
    validate_sequence(actions)
    for action in actions:
        if isinstance(action, str):
            if action not in SIMPLE_ACTION_TYPES:
                raise ConfigSyntaxError(f"Unknown action keyword `{action}` must be from {SIMPLE_ACTION_TYPES}")
        elif isinstance(action, dict):
            try:
                validate_if_present(action, STEPS_NUMBER_KEY, validate_positive_int)
                validate_if_present(action, WEIGHT_KEY, validate_0_to_1_number)
                validate_if_present(action, INITIAL_MIXING_KEY, validate_0_to_1_number)
                validate_if_present(action, FINAL_MIXING_KEY, validate_0_to_1_number)
            except ConfigSyntaxError as e:
                raise ConfigSyntaxError("Invalid action") from e
        else:
            raise ConfigSyntaxError(f"Unknown action type `{type(action)}` must be a dict or a string from {SIMPLE_ACTION_TYPES}")
    qa_actions = [action for action in actions if isinstance(action, dict)]
    if qa_actions:
        total_weight = sum(action.get(WEIGHT_KEY, DEFAULT_WEIGHT) for action in qa_actions)
        if not isclose(total_weight, 1.):
            raise ConfigSyntaxError(f"Total weight aggregated among all the actions must be close to 1.0, but actual total weight is {total_weight}")


def validate_schedule(sch):
    if not isinstance(sch, dict):
        raise ConfigSyntaxError(f"Must be a dictionary, but received a value of type {type(sch)}")
    validate_if_present(sch, TOTAL_TIME_KEY, validate_positive_number)
    validate_if_present(sch, STARTING_MIXING_KEY, validate_0_to_1_number)
    validate_if_present(sch, ACTIONS_KEY, validate_actions)


def validate_sparsification(sp):
    if sp is None:
        return
    if not isinstance(sp, dict):
        raise ConfigSyntaxError(f"Must be a dictionary, but received a value of type {type(sp)}")
    validate_if_present(sp, EPS_KEY, validate_0_to_1_number)
    validate_if_present(sp, CLUSTER_COUPLING_AMPLITUDE_KEY, validate_number_1_to_inf)


def validate_sparsification_actions_consistency(config):
    if config.get(SPARSIFICATION_KEY) is not None \
       and SCHEDULE_KEY in config \
       and ACTIONS_KEY in config[SCHEDULE_KEY] \
       and any(action == GET_BLOCH_VECTORS for action in config[SCHEDULE_KEY][ACTIONS_KEY]):
        raise ConfigSyntaxError(f"Cannot use `{GET_BLOCH_VECTORS}` action if `{SPARSIFICATION_KEY}` is enabled")


def validate_config(config):
    if not isinstance(config, dict):
        raise ConfigSyntaxError(f"Config must be a dictionary, but `{type(config)}` is received")
    if EDGES_KEY not in config:
        raise ConfigSyntaxError(f"Invalid config: `{EDGES_KEY}` is not present")
    validate_edges(config[EDGES_KEY])
    validate_all_records(
        config,
        [
            [NODES_KEY, validate_nodes],
            [DEFAULT_FIELD_KEY, validate_number],
            [SCHEDULE_KEY, validate_schedule],
            [MAX_BOND_DIM_KEY, validate_positive_int],
            [MAX_BP_ITER_NUMBER_KEY, validate_non_neg_int],
            [SEED_KEY, validate_non_neg_int],
            [BP_EPS_KEY, validate_0_to_1_number],
            [PINV_EPS_KEY, validate_0_to_1_number],
            [MEASUREMENT_THRESHOLD_KEY, validate_half_to_1_number],
            [DAMPING_KEY, validate_0_to_1_number],
            [BACKEND_KEY, validate_backend],
            [SPARSIFICATION_KEY, validate_sparsification]
        ],
    )
    validate_sparsification_actions_consistency(config)
    return config

