import logging
from itertools import chain
from random import Random
from bqa import run_qa

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def get_rand_ampl(rng: Random) -> int:
    return 2 * rng.randint(0, 1) - 1

def get_full_size_heavy_hex_edges_with_random_ampls(rng: Random) -> dict[tuple[int, int], float]:
    edges = {}
    for i in chain(range(13),
                   range(18, 32),
                   range(37, 51),
                   range(56, 70),
                   range(75, 89),
                   range(94, 108),
                   range(113, 126)):
        edges[(i, i + 1)] = get_rand_ampl(rng)
    edges[(0, 14)] = get_rand_ampl(rng)
    edges[(14, 18)] = get_rand_ampl(rng)
    edges[(4, 15)] = get_rand_ampl(rng)
    edges[(15, 22)] = get_rand_ampl(rng)
    edges[(8, 16)] = get_rand_ampl(rng)
    edges[(16, 26)] = get_rand_ampl(rng)
    edges[(12, 17)] = get_rand_ampl(rng)
    edges[(17, 30)] = get_rand_ampl(rng)
    edges[(20, 33)] = get_rand_ampl(rng)
    edges[(33, 39)] = get_rand_ampl(rng)
    edges[(24, 34)] = get_rand_ampl(rng)
    edges[(34, 43)] = get_rand_ampl(rng)
    edges[(28, 35)] = get_rand_ampl(rng)
    edges[(35, 47)] = get_rand_ampl(rng)
    edges[(32, 36)] = get_rand_ampl(rng)
    edges[(36, 51)] = get_rand_ampl(rng)
    edges[(37, 52)] = get_rand_ampl(rng)
    edges[(52, 56)] = get_rand_ampl(rng)
    edges[(41, 53)] = get_rand_ampl(rng)
    edges[(53, 60)] = get_rand_ampl(rng)
    edges[(45, 54)] = get_rand_ampl(rng)
    edges[(54, 64)] = get_rand_ampl(rng)
    edges[(49, 55)] = get_rand_ampl(rng)
    edges[(55, 68)] = get_rand_ampl(rng)
    edges[(58, 71)] = get_rand_ampl(rng)
    edges[(71, 77)] = get_rand_ampl(rng)
    edges[(62, 72)] = get_rand_ampl(rng)
    edges[(72, 81)] = get_rand_ampl(rng)
    edges[(66, 73)] = get_rand_ampl(rng)
    edges[(73, 85)] = get_rand_ampl(rng)
    edges[(70, 74)] = get_rand_ampl(rng)
    edges[(74, 89)] = get_rand_ampl(rng)
    edges[(75, 90)] = get_rand_ampl(rng)
    edges[(90, 94)] = get_rand_ampl(rng)
    edges[(79, 91)] = get_rand_ampl(rng)
    edges[(91, 98)] = get_rand_ampl(rng)
    edges[(83, 92)] = get_rand_ampl(rng)
    edges[(92, 102)] = get_rand_ampl(rng)
    edges[(87, 93)] = get_rand_ampl(rng)
    edges[(93, 106)] = get_rand_ampl(rng)
    edges[(96, 109)] = get_rand_ampl(rng)
    edges[(109, 114)] = get_rand_ampl(rng)
    edges[(100, 110)] = get_rand_ampl(rng)
    edges[(110, 118)] = get_rand_ampl(rng)
    edges[(104, 111)] = get_rand_ampl(rng)
    edges[(111, 122)] = get_rand_ampl(rng)
    edges[(108, 112)] = get_rand_ampl(rng)
    edges[(112, 126)] = get_rand_ampl(rng)
    return edges

def get_full_size_heavy_hex_nodes_with_random_ampls(rng: Random) -> dict[int, float]:
    return {node_id : get_rand_ampl(rng) for node_id in range(127)}

rng = Random(42)

nodes = get_full_size_heavy_hex_nodes_with_random_ampls(rng)
edges = get_full_size_heavy_hex_edges_with_random_ampls(rng)

config = {
    "nodes" : nodes,
    "edges" : edges,
    "max_bond_dim" : 8,
    "backend" : "numpy",
    "schedule" : {
        "total_time" : 10.,
        "starting_mixing" : 1.,
        "actions" : [
            {
                "weight" : 1.,
                "steps_number" : 10,
                "final_mixing" : 0.0,
            },
            "measure",
        ]
    },
}

bptn_result = run_qa(config)

print(bptn_result)
