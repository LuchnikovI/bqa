import logging
from bqa import run_qa
from bqa.benchmarking import generate_qubo_on_2d_grid

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

nodes, edges = generate_qubo_on_2d_grid(20, 20)

config = {
    "nodes" : nodes,
    "edges" : edges,
    "max_bond_dim" : 4,
    "backend" : "numpy",
    "schedule" : {
        "total_time" : 10.,
        "starting_mixing" : 1.,
        "actions" : [
            {
                "weight" : 1.,
                "steps_number" : 100,
                "final_mixing" : 0.0,
            },
            "measure",
        ]
    },
}

bptn_result = run_qa(config)

print(bptn_result)
