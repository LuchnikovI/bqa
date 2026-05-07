from textwrap import indent
from bqa.benchmarking import generate_qubo_on_2d_grid, generate_qubo_on_random_regular_graph
from bqa.config.core import canonicalize, full_preprocess, get_metrics
from bqa.config.desugar_config import (DEFAULT_CLUSTER_COUPLING_AMPLITUDE, DEFAULT_EPS, DEFAULT_MAX_BOND_DIM,
                                       DEFAULT_MAX_BP_ITERS_NUMBER, DEFAULT_TOTAL_TIME, desug_or_warn_and_set_default_if_not_present,
                                       desugared_config_to_json)
from bqa.core import run_qa
from bqa.cli_utils import JSONInputOutputCli, json_input_output_cli
from bqa.config.validate_config import (ACTIONS_KEY, BACKEND_KEY, CLUSTER_COUPLING_AMPLITUDE_KEY, DAMPING_KEY, EDGES_KEY, EPS_KEY,
                                        FINAL_MIXING_KEY, INITIAL_MIXING_KEY, MAX_BOND_DIM_KEY, MAX_BP_ITER_NUMBER_KEY,
                                        MEASURE, NODES_KEY, SCHEDULE_KEY, SEED_KEY, SPARSIFICATION_KEY, STARTING_MIXING_KEY, STEPS_NUMBER_KEY,
                                        TOTAL_TIME_KEY, WEIGHT_KEY, ConfigSyntaxError, validate_all_records, validate_config, validate_non_neg_int,
                                        validate_number, validate_positive_int)


def _list_qa_clis():
    print("""There are following quantum annealing oriented CLI tools:""")
    for name, doc in JSONInputOutputCli.registered_clis.items():
        print(f"\n{name[1:]}:")
        print(indent(doc, "\t"))


@json_input_output_cli(doc = run_qa.__doc__)
def _run_qa(config):
    return run_qa(config)


@json_input_output_cli(doc = validate_config.__doc__)
def _validate(config):
    return validate_config(config)


@json_input_output_cli(doc = canonicalize.__doc__)
def _canonicalize(config):
    return desugared_config_to_json(canonicalize(config))


@json_input_output_cli(doc = full_preprocess.__doc__)
def _full_preprocess(config):
    return desugared_config_to_json(full_preprocess(config))


def validate_degree(degree):
    if not isinstance(degree, int):
        raise ConfigSyntaxError(f"Must be integer value, got value of type `{type(degree)}`")
    if degree < 3:
        raise ConfigSyntaxError(f"Must be >= 3, but got value {degree}")


@json_input_output_cli(
    doc = """Generates a json config for an optimization problem on a random regular graph with
random magnetic fields and couplings sampled uniformly from a fixed interval.
An example of an input json config:
     {
         "degree" : 4,  # degree of a graph (default 3)
         "nodes_number" : 1500,  # number of nodes in a graph (default 100)
         "seed" :   42,   # random seed (default 42)
         "j_max" :  1.0,  # maximal coupling value (default 1.0)
         "j_min" : -1.0,  # minimal coupling value (default 1.0)
         "h_max" :  1.0,  # maximal magnetic field value (default 0.0)
         "h_min" : -1.0   # minimal magnetic field value (default 0.0)
     },
All the fields in json are optional and set to default values if not present."""
)
def _random_regular_graph(config):
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


@json_input_output_cli(
    doc = """Generates a json config for an optimization problem on a 2d grid with
random magnetic fields and couplings sampled uniformly from a fixed interval.
An example of an input json config:
     {
         "rows" : 50,  # number of rows in a grid (default 10)
         "cols" : 60,  # number of columns in a grid (default 10)
         "seed" :   42,   # random seed (default 42)
         "j_max" :  1.0,  # maximal coupling value (default 1.0)
         "j_min" : -1.0,  # minimal coupling value (default 1.0)
         "h_max" :  1.0,  # maximal magnetic field value (default 0.0)
         "h_min" : -1.0   # minimal magnetic field value (default 0.0)
    },
All the fields in json are optional and set to default values if not present."""
)
def _2d_grid(config):
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


def validate_dict(config, name):
    if not isinstance(config, dict):
        raise ConfigSyntaxError(f"Input {name} must be a dictionary, but recieved `{type(config)}`")


@json_input_output_cli(doc = "Sets backend to `cupy`.")
def _cupy(config):
    validate_dict(config, "config")
    config[BACKEND_KEY] = "cupy"
    return config


@json_input_output_cli(doc = "Sets default sparsification strategy.")
def _sparsify(config):
    validate_dict(config, "config")
    config[SPARSIFICATION_KEY] = {EPS_KEY : DEFAULT_EPS, CLUSTER_COUPLING_AMPLITUDE_KEY : DEFAULT_CLUSTER_COUPLING_AMPLITUDE}
    return config


@json_input_output_cli(doc = get_metrics.__doc__)
def _metrics(config):
    return get_metrics(config)


@json_input_output_cli(
    doc = """Sets linear quantum annealing schedule according the metrics of a given optimization problem
after full preprocessing (validation -> desugaring -> sparsification)."""
)
def _adjust_schedule(config):
    metrics = get_metrics(config)
    mean_degree = metrics["mean_degree"]
    mean_abs_coupling = metrics["mean_abs_coupling"]
    mean_abs_field = metrics["mean_abs_field"]
    total_time = 20 * mean_degree / mean_abs_coupling
    dt = 1 / (10 * (mean_degree * mean_abs_coupling + mean_abs_field))
    steps_number = int(total_time / dt) + 1
    config[SCHEDULE_KEY] = {
        TOTAL_TIME_KEY : total_time,
        STARTING_MIXING_KEY : 1.0,
        ACTIONS_KEY : [
            {
                INITIAL_MIXING_KEY : 1.0,
                FINAL_MIXING_KEY : 0.0,
                WEIGHT_KEY : 1.0,
                STEPS_NUMBER_KEY : steps_number,
            },
            MEASURE,
        ]
    }
    return desugared_config_to_json(config)


@json_input_output_cli(
    doc = f"Sets `{MAX_BOND_DIM_KEY}` twice larger."
)
def _x2_bond_dim(config):
    validate_dict(config, "config")
    config[MAX_BOND_DIM_KEY] = 2 * config.get(MAX_BOND_DIM_KEY, DEFAULT_MAX_BOND_DIM)
    return config


@json_input_output_cli(
    doc = f"""Sets new value of damping as follows `{DAMPING_KEY} <- 0.5 * {DAMPING_KEY} + 0.5`,
i.e. 0 -> 0.5, 0.5 -> 0.75, etc."""
)
def _inc_damping(config):
    validate_dict(config, "config")
    prev_damping = config.get(DAMPING_KEY, 0.)
    config[DAMPING_KEY] = 0.5 * prev_damping + 0.5
    return config


@json_input_output_cli(doc = f"Sets `{MAX_BP_ITER_NUMBER_KEY}` twice larger.")
def _x2_bp_iters(config):
    validate_dict(config, "config")
    config[MAX_BP_ITER_NUMBER_KEY] = 2 * config.get(MAX_BP_ITER_NUMBER_KEY, DEFAULT_MAX_BP_ITERS_NUMBER)
    return config


@json_input_output_cli(doc = f"Sets `{TOTAL_TIME_KEY}` twice larger.")
def _x2_total_time(config):
    validate_dict(config, "config")
    if SCHEDULE_KEY in config:
        schedule = config[SCHEDULE_KEY]
        validate_dict(schedule, "schedule")
        schedule[TOTAL_TIME_KEY] = 2 * schedule.get(TOTAL_TIME_KEY, DEFAULT_TOTAL_TIME)
    else:
        config[SCHEDULE_KEY] = {TOTAL_TIME_KEY : 2 * DEFAULT_TOTAL_TIME}
    return config

