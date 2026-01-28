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
            "measure",
        ]
    },
    "runtime_limit" : 10,
    "seed" : 42,
}

run_benchmarks(config)

