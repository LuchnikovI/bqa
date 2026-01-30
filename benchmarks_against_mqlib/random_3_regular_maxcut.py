import logging
from utils import run_benchmarks
from bqa.benchmarking import generate_qubo_on_random_regular_graph

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# parameters

nodes_number = 1000
nodes, edges = generate_qubo_on_random_regular_graph(
    nodes_number,
    node_ampl_func=lambda _, x: 0.,
    edge_ampl_func=lambda _, x: 1.,
)
config = {
    "description" : f"This is a MaxCut problem with {nodes} nodes on a 3-regular graph. Note, that one minimizes the Ising Hamiltonian, so the resulting energy must be shifted to get the cut size.",
    "experiment_name" : "random_3_regular_maxcut",
    "nodes" : nodes,
    "edges" : edges,
    "max_bond_dim" : 10,
    "measurement_threshold" : 0.99,
    "damping" : 0.5,
    "bp_eps" : 1e-5,
    "pinv_eps" : 1e-5,
    "max_bp_iter_number" : 250,
    "backend" : "cupy",
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
    "runtime_limit" : 10,  # MQLib heuristics runtime limit
    "seed" : 42,
}

run_benchmarks(config)

