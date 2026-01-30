import logging
from utils import run_benchmarks
from bqa.benchmarking import generate_qubo_on_random_regular_graph

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# parameters

nodes_number = 100000
nodes, edges = nodes, edges = generate_qubo_on_random_regular_graph(nodes_number)
config = {
    "description" : f"This is a QUBO problem with {nodes_number} on a 3-regular graph with couplings and local fields sampled from uniform(-1, 1).",
    "experiment_name" : "random_3_regular_random_fields_random_interactions",
    "nodes" : nodes,
    "edges" : edges,
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

