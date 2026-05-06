from bqa.benchmarking import generate_qubo_on_2d_grid, generate_qubo_on_random_regular_graph
from bqa.config.core import canonicalize, full_preprocess, get_metrics
from bqa.config.desugar_config import (DEFAULT_CLUSTER_COUPLING_AMPLITUDE, DEFAULT_EPS, DEFAULT_MAX_BOND_DIM,
                                       DEFAULT_MAX_BP_ITERS_NUMBER, desug_or_warn_and_set_default_if_not_present,
                                       desugared_config_to_json)
from bqa.core import run_qa
from bqa.cli_utils import json_input_output_cli
from bqa.config.validate_config import (ACTIONS_KEY, BACKEND_KEY, CLUSTER_COUPLING_AMPLITUDE_KEY, DAMPING_KEY, EDGES_KEY, EPS_KEY,
                                        FINAL_MIXING_KEY, GET_BLOCH_VECTORS, INITIAL_MIXING_KEY, MAX_BOND_DIM_KEY, MAX_BP_ITER_NUMBER_KEY,
                                        MEASURE, NODES_KEY, SCHEDULE_KEY, SEED_KEY, SPARSIFICATION_KEY, STARTING_MIXING_KEY, STEPS_NUMBER_KEY,
                                        TOTAL_TIME_KEY, WEIGHT_KEY, ConfigSyntaxError, validate_all_records, validate_config, validate_if_present, validate_non_neg_int, validate_number, validate_positive_int)


@json_input_output_cli
def _validate(config):
    "Config validator CLI. Use it to check the correctness of the config without execution."
    return validate_config(config)


@json_input_output_cli
def _run_qa(config):
    "Quantum annealing simulation runner CLI."
    return run_qa(config)


@json_input_output_cli
def _canonicalize(config):
    "Config canonicalization CLI. Use it to see how the config looks after the canonicalization but before sparsification."
    return desugared_config_to_json(canonicalize(config))


@json_input_output_cli
def _full_preprocess(config):
    "Full config preprocessing CLI. Use it to see how the config looks after the canonicalization and sparsification, right before the execution."
    return desugared_config_to_json(full_preprocess(config))

def validate_degree(degree):
    if not isinstance(degree, int):
        raise ConfigSyntaxError(f"Must be integer value, got value of type `{type(degree)}`")
    if degree < 3:
        raise ConfigSyntaxError(f"Must be >= 3, but got value {degree}")


# TODO: make propper input data validation
@json_input_output_cli
def _random_regular_graph(config):
    "Generates an optimization problem on a random regular graph."
    if not isinstance(config, dict):
        raise ConfigSyntaxError(f"Input config to the random regular graph generator must be a dictionary, but recived {type(config)}")
    validate_all_records(
        config,
        [
            ["degree", validate_degree],
            ["nodes_number", validate_positive_int],
            ["seed", validate_non_neg_int],
            ["j_max", validate_number],
            ["j_min", validate_number],
            ["h_max", validate_number],
            ["h_min", validate_number],
        ]
    )
    degree = desug_or_warn_and_set_default_if_not_present(config, "degree", 3, int)
    nodes_number = desug_or_warn_and_set_default_if_not_present(config, "nodes_number", 100, int)
    seed = desug_or_warn_and_set_default_if_not_present(config, "seed", 42, int)
    j_max = desug_or_warn_and_set_default_if_not_present(config, "j_max", 1.0, float)
    j_min = desug_or_warn_and_set_default_if_not_present(config, "j_min", 1.0, float)
    h_max = desug_or_warn_and_set_default_if_not_present(config, "h_max", 0.0, float)
    h_min = desug_or_warn_and_set_default_if_not_present(config, "h_min", 0.0, float)
    nodes, edges = generate_qubo_on_random_regular_graph(
        nodes_number,
        degree,
        seed,
        lambda rng, _: rng.uniform(h_min, h_max),
        lambda rng, _: rng.uniform(j_min, j_max),
    )
    return desugared_config_to_json({EDGES_KEY : edges, NODES_KEY : nodes, SEED_KEY : seed})


# TODO: make propper input data validation
@json_input_output_cli
def _2d_grid(config):
    "Generates and optimization problem on a 2D grid."
    if not isinstance(config, dict):
        raise ValueError(f"Input config to the random regular graph generator must be a dictionary, but recived {type(config)}")
    validate_all_records(
        config,
        [
            ["rows", validate_positive_int],
            ["cols", validate_positive_int],
            ["seed", validate_non_neg_int],
            ["j_max", validate_number],
            ["j_min", validate_number],
            ["h_max", validate_number],
            ["h_min", validate_number],
        ]
    )
    rows = desug_or_warn_and_set_default_if_not_present(config, "rows", 10, int)
    cols = desug_or_warn_and_set_default_if_not_present(config, "cols", 10, int)
    seed = desug_or_warn_and_set_default_if_not_present(config, "seed", 42, int)
    j_max = desug_or_warn_and_set_default_if_not_present(config, "j_max", 1.0, float)
    j_min = desug_or_warn_and_set_default_if_not_present(config, "j_min", 1.0, float)
    h_max = desug_or_warn_and_set_default_if_not_present(config, "h_max", 0.0, float)
    h_min = desug_or_warn_and_set_default_if_not_present(config, "h_min", 0.0, float)
    nodes, edges = generate_qubo_on_2d_grid(
        rows,
        cols,
        seed,
        lambda rng, _: rng.uniform(h_min, h_max),
        lambda rng, _: rng.uniform(j_min, j_max),
    )
    return desugared_config_to_json({EDGES_KEY : edges, NODES_KEY : nodes, SEED_KEY : seed})


@json_input_output_cli
def _cupy(config):
    "Sets backend to `cupy`."
    config[BACKEND_KEY] = "cupy"
    return config


@json_input_output_cli
def _sparsify(config):
    "Sets default sparsification strategy."
    config[SPARSIFICATION_KEY] = {EPS_KEY : DEFAULT_EPS, CLUSTER_COUPLING_AMPLITUDE_KEY : DEFAULT_CLUSTER_COUPLING_AMPLITUDE}
    return config


@json_input_output_cli
def _metrics(config):
    "Returns some metrics of the optimization problem after full preprocessing."
    return get_metrics(config)


@json_input_output_cli
def _adjust_schedule(config):
    "Sets quantum annealing schedule according the problem's metrics."
    config = full_preprocess(config)
    metrics = get_metrics(config)
    total_time = 20 * metrics["mean_degree"] / metrics["mean_abs_coupling"]
    steps_number = 10 * total_time * (metrics["mean_degree"] * metrics["mean_abs_coupling"] + metrics["mean_abs_field"])
    config[SCHEDULE_KEY] = {
        TOTAL_TIME_KEY : total_time,
        STARTING_MIXING_KEY : 1.0,
        ACTIONS_KEY : [
            {
                INITIAL_MIXING_KEY : 1.0,
                FINAL_MIXING_KEY : 0.0,
                WEIGHT_KEY : 1.0,
                STEPS_NUMBER_KEY : int(steps_number),
            },
            MEASURE,
        ]
    }
    return desugared_config_to_json(config)


@json_input_output_cli
def _x2_bond_dim(config):
    f"Sets `{MAX_BOND_DIM_KEY}` twice larger."
    if MAX_BOND_DIM_KEY in config:
        config[MAX_BOND_DIM_KEY] *= 2
    else:
        config[MAX_BOND_DIM_KEY] = 2 * DEFAULT_MAX_BOND_DIM
    return config


@json_input_output_cli
def _inc_damping(config):
    f"Sets new value of damping as follows `{DAMPING_KEY} <- 0.5 * {DAMPING_KEY} + 0.5`."
    prev_damping = config.get(DAMPING_KEY, 0.)
    config[DAMPING_KEY] = 0.5 * prev_damping + 0.5
    return config


@json_input_output_cli
def _x2_bp_iters(config):
    f"Sets {MAX_BP_ITER_NUMBER_KEY} twice larger."
    if MAX_BP_ITER_NUMBER_KEY in config:
        config[MAX_BOND_DIM_KEY] *= 2
    else:
        config[MAX_BOND_DIM_KEY] = 2 * DEFAULT_MAX_BP_ITERS_NUMBER
    return config

