import logging
from utils import run_benchmarks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

config = {
    "description" : "This is a MaxCut problem with 1000 nodes on a 10-regular graph. Note, that one minimizes the Ising Hamiltonian, so the resulting energy must be shifted to get the cut size.",
    "experiment_name" : "random_10_regular_maxcut_1000",
    "generator_function_name" : "generate_qubo_on_random_regular_graph",
    "args" : {
        "nodes_number" : 1000,  # key can be any, serves as documentation
        "degree" : 10,
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
    "backend" : "cupy",
    "schedule" : {
        "total_time" : 100.,
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
    "runtime_limit" : 100,  # MQLib heuristics runtime limit
    "hard_runtime_limit" : 100,
    "seed" : 42,
    "sparsification" : {}
}

run_benchmarks(config)

