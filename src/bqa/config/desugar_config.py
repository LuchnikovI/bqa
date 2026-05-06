from copy import deepcopy
import logging

from bqa.config.validate_config import (
    ACTIONS_KEY,
    CLUSTER_COUPLING_AMPLITUDE_KEY,
    EPS_KEY,
    FINAL_MIXING_KEY,
    GET_BLOCH_VECTORS,
    INITIAL_MIXING_KEY,
    MEASURE,
    NODES_KEY,
    EDGES_KEY,
    MAX_BOND_DIM_KEY,
    MAX_BP_ITER_NUMBER_KEY,
    BP_EPS_KEY,
    PINV_EPS_KEY,
    BACKEND_KEY,
    DEFAULT_FIELD_KEY,
    MEASUREMENT_THRESHOLD_KEY,
    SCHEDULE_KEY,
    SEED_KEY,
    DAMPING_KEY,
    SPARSIFICATION_KEY,
    STARTING_MIXING_KEY,
    STEPS_NUMBER_KEY,
    TOTAL_TIME_KEY,
    VALIDATED_KEY,
    WEIGHT_KEY,
)
from bqa.config.pipeline import pipeline

# defaults

DEFAULT_NODES = {}

DEFAULT_MAX_BOND_DIM = 4

DEFAULT_MAX_BP_ITERS_NUMBER = 75

DEFAULT_BP_EPS = 1e-5

DEFAULT_PINV_EPS = 1e-5

DEFAULT_BACKEND = "numpy"

DEFAULT_DEFAULT_FIELD = 0.0

DEFAULT_MEASUREMENT_THRESHOLD = 0.99

DEFAULT_SEED = 42

DEFAULT_DAMPING = 0.

DEFAULT_SPARSIFICATION = None

DEFAULT_POSTPROCESSING = None

DEFAULT_TOTAL_TIME = 10.0

DEFAULT_STEPS_NUMBER = 100

DEFAULT_WEIGHT = 1.0

DEFAULT_STARTING_MIXING = 1.0

DEFAULT_FINAL_MIXING = 0.0

DEFAULT_CLUSTER_COUPLING_AMPLITUDE = 1.1

DEFAULT_EPS = 0.0

DEFAULT_EVOLUTION = {
    WEIGHT_KEY : DEFAULT_WEIGHT,
    STEPS_NUMBER_KEY : DEFAULT_STEPS_NUMBER,
    INITIAL_MIXING_KEY : DEFAULT_STARTING_MIXING,
    FINAL_MIXING_KEY : DEFAULT_FINAL_MIXING,
}

SIMPLE_ACTIONS = {MEASURE, GET_BLOCH_VECTORS}

DEFAULT_ACTIONS = [
    DEFAULT_EVOLUTION,
    MEASURE,
]

DEFAULT_SCHEDULE = {
    TOTAL_TIME_KEY : DEFAULT_TOTAL_TIME,
    STARTING_MIXING_KEY : DEFAULT_STARTING_MIXING,
    ACTIONS_KEY : DEFAULT_ACTIONS,
}

TOP_DESUGARED_KEYS = {NODES_KEY, EDGES_KEY, MAX_BOND_DIM_KEY, MAX_BP_ITER_NUMBER_KEY, BP_EPS_KEY, PINV_EPS_KEY,
                      BACKEND_KEY, DEFAULT_FIELD_KEY, MEASUREMENT_THRESHOLD_KEY, SEED_KEY, DAMPING_KEY, SCHEDULE_KEY,
                      SPARSIFICATION_KEY}

DESUGARED_KEY = "desugared"

log = logging.getLogger(__name__)


def get_iter(container):
    return map(tuple, (container.items() if isinstance(container, dict) else container))


def get_ids_iter(container):
    return map(lambda x: x[0], get_iter(container))


def canonicalize_edge_id(edge_id):
    i, j = edge_id
    assert i != j
    return (i, j) if i < j else (j, i)


def desug_or_warn_and_set_default_if_not_present(config, key, default_value, desug_fn=None, *args):
    if key in config:
        if desug_fn is None:
            return config[key]
        else:
            return desug_fn(config[key], *args)
    else:
        log.warning(f"Set `{key}` to default value {default_value}")
        return deepcopy(default_value) if isinstance(default_value, (list, dict)) else default_value


def desug_sparsification(sparsification):
    if sparsification is None:
        return None
    return {
        EPS_KEY : desug_or_warn_and_set_default_if_not_present(
            sparsification,
            EPS_KEY,
            DEFAULT_EPS,
            float,
        ),
        CLUSTER_COUPLING_AMPLITUDE_KEY : desug_or_warn_and_set_default_if_not_present(
            sparsification,
            CLUSTER_COUPLING_AMPLITUDE_KEY,
            DEFAULT_CLUSTER_COUPLING_AMPLITUDE,
            float,
        ),
    }


def desug_edges(edges):
    return {canonicalize_edge_id(edge_id) : float(cpl) \
            for edge_id, cpl \
            in (edges.items() if isinstance(edges, dict) else edges)}


def desug_nodes(nodes):
    return {node_id : float(ampl) for node_id, ampl in (nodes.items() if isinstance(nodes, dict) else nodes)}


def desug_actions_seq(actions, starting_mixing):
    mixing = starting_mixing
    for action in actions:
        if isinstance(action, str):
            yield action
        else:
            desug_action = {
                STEPS_NUMBER_KEY : desug_or_warn_and_set_default_if_not_present(action, STEPS_NUMBER_KEY, DEFAULT_STEPS_NUMBER),
                WEIGHT_KEY : float(action[WEIGHT_KEY]),
                INITIAL_MIXING_KEY : desug_or_warn_and_set_default_if_not_present(action, INITIAL_MIXING_KEY, mixing, float),
                FINAL_MIXING_KEY : desug_or_warn_and_set_default_if_not_present(action, FINAL_MIXING_KEY, mixing, float),
            }
            mixing = desug_action[FINAL_MIXING_KEY]
            yield desug_action


def desug_actions(actions, starting_mixing):
    return list(desug_actions_seq(actions, starting_mixing))


def desug_schedule(schedule):
    starting_mixing = desug_or_warn_and_set_default_if_not_present(schedule, STARTING_MIXING_KEY, DEFAULT_STARTING_MIXING, float)
    return {
        TOTAL_TIME_KEY : desug_or_warn_and_set_default_if_not_present(schedule, TOTAL_TIME_KEY, DEFAULT_TOTAL_TIME, float),
        STARTING_MIXING_KEY : starting_mixing,
        ACTIONS_KEY : desug_or_warn_and_set_default_if_not_present(schedule, ACTIONS_KEY, DEFAULT_ACTIONS, desug_actions, starting_mixing),
    }


@pipeline
def desugar_config(config):
    assert config.get(VALIDATED_KEY), "`desugar_config` call on non validated config"
    if config.get(DESUGARED_KEY):
        log.warning(f"`{DESUGARED_KEY}` flag set to `{config[DESUGARED_KEY]}`, skipping desugaring")
        return config
    return {
        NODES_KEY : desug_or_warn_and_set_default_if_not_present(config, NODES_KEY, DEFAULT_NODES, desug_nodes),
        EDGES_KEY : desug_edges(config[EDGES_KEY]),
        MAX_BOND_DIM_KEY : desug_or_warn_and_set_default_if_not_present(config, MAX_BOND_DIM_KEY, DEFAULT_MAX_BOND_DIM),
        MAX_BP_ITER_NUMBER_KEY : desug_or_warn_and_set_default_if_not_present(config, MAX_BP_ITER_NUMBER_KEY, DEFAULT_MAX_BP_ITERS_NUMBER),
        BP_EPS_KEY : desug_or_warn_and_set_default_if_not_present(config, BP_EPS_KEY, DEFAULT_BP_EPS, float),
        PINV_EPS_KEY : desug_or_warn_and_set_default_if_not_present(config, PINV_EPS_KEY, DEFAULT_PINV_EPS, float),
        BACKEND_KEY : desug_or_warn_and_set_default_if_not_present(config, BACKEND_KEY, DEFAULT_BACKEND),
        DEFAULT_FIELD_KEY : desug_or_warn_and_set_default_if_not_present(config, DEFAULT_FIELD_KEY, DEFAULT_DEFAULT_FIELD, float),
        MEASUREMENT_THRESHOLD_KEY : desug_or_warn_and_set_default_if_not_present(
            config,
            MEASUREMENT_THRESHOLD_KEY,
            DEFAULT_MEASUREMENT_THRESHOLD,
            float,
        ),
        SEED_KEY : desug_or_warn_and_set_default_if_not_present(config, SEED_KEY, DEFAULT_SEED),
        DAMPING_KEY : desug_or_warn_and_set_default_if_not_present(config, DAMPING_KEY, DEFAULT_DAMPING, float),
        SCHEDULE_KEY : desug_or_warn_and_set_default_if_not_present(config, SCHEDULE_KEY, DEFAULT_SCHEDULE, desug_schedule),
        SPARSIFICATION_KEY : desug_or_warn_and_set_default_if_not_present(
            config,
            SPARSIFICATION_KEY,
            DEFAULT_SPARSIFICATION,
            desug_sparsification,
        ),
        DESUGARED_KEY : True,
        **{k : v for k, v in config.items() if k not in TOP_DESUGARED_KEYS},
    }


def desugared_config_to_json(config):
    edges = config[EDGES_KEY]
    nodes = config[NODES_KEY]
    edges = [[list(edge_id), cpl] for edge_id, cpl in get_iter(edges)]
    nodes = [[node_id, field] for node_id, field in get_iter(nodes)]
    return {EDGES_KEY : edges, NODES_KEY : nodes, **{k : v for k, v in config.items() if k not in {NODES_KEY, EDGES_KEY}}}

