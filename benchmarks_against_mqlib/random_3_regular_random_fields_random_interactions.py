import logging
from utils import run_benchmarks

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

config = {
    "description" : "This is a QUBO problem with 100000 on a 3-regular graph with couplings and local fields sampled from uniform(-1, 1).",
    "experiment_name" : "random_3_regular_random_fields_random_interactions",
    "generator_function_name" : "generate_qubo_on_random_regular_graph",
    "args" : {
        "nodes_number" : 100000,  # key can be any, serves as documentation
    },
    "max_bond_dim" : 4,
    "backend" : "numpy",
    "schedule" : {
        "total_time" : 2000.,
        "starting_mixing" : 1.,
        "actions" : [
            {
                "weight" : 1.,
                "steps_number" : 10000,
                "final_mixing" : 0.0,
            },
            "get_bloch_vectors",  # one uses Bloch vectors to reconstruct solution since measurements sampling is too long
        ]
    },
    "runtime_limit" : 10,  # MQLib heuristics runtime limit
    "seed" : 42,
}

run_benchmarks(config)

