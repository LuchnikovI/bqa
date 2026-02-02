import logging
from utils import run_benchmarks

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

config = {
    "description" : "This is a MaxCut problem with 1000 nodes on a 3-regular graph. Note, that one minimizes the Ising Hamiltonian, so the resulting energy must be shifted to get the cut size.",
    "experiment_name" : "random_3_regular_maxcut",
    "generator_function_name" : "generate_qubo_on_random_regular_graph",
    "args" : {
        "nodes_number" : 1000,  # key can be any, serves as documentation
        "degree" : 3,
    },
    "kwargs" : {
        "node_ampl_func" : lambda _0, _1: 0.,
        "edge_ampl_func" : lambda _0, _1: 1.,
    },
    "max_bond_dim" : 16,
    "measurement_threshold" : 0.99,
    "damping" : 0.5,
    "bp_eps" : 1e-5,
    "pinv_eps" : 1e-5,
    "max_bp_iter_number" : 250,
    "backend" : "numpy",
    "schedule" : {
        "total_time" : 200.,
        "starting_mixing" : 1.,
        "actions" : [
            {
                "weight" : 1.,
                "steps_number" : 1000,
                "final_mixing" : 0.,
            },
            "measure",
        ]
    },
    "runtime_limit" : 1,  # MQLib heuristics runtime limit
    "seed" : [42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
}

run_benchmarks(config)

