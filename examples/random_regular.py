import logging
from bqa import run_qa
from bqa.benchmarking import generate_qubo_on_random_regular_graph

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

nodes, edges = generate_qubo_on_random_regular_graph(1000)

config = {
    "nodes" : nodes,
    "edges" : edges,
    "max_bond_dim" : 4,
    "backend" : "numpy",
    "schedule" : {
        "total_time" : 100.,
        "starting_mixing" : 1.,
        "actions" : [
            {
                "weight" : 1.,
                "steps_number" : 1000,
                "final_mixing" : 0.0,
            },
            "measure",
        ]
    },
    "runtime_limit" : 1,
}

bptn_result = run_qa(config)

print(bptn_result)
