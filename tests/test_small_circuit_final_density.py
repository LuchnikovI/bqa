import numpy as np
from bqa.core import run_qa
from .exact_sim import run_qa_exact

CONFIG = {
    "nodes" : {0 : 1., 1 : -1., 2 : 0.5, 3 : -0.5, 4: 1.1, 5 : 0.4},
    "edges" : {(0, 1) : 1., (2, 1) : -1., (1, 3) : 1., (4, 3) : -1., (5, 3) : 1.},
    "pinv_eps" : 1e-9,
    "bp_eps" : 1e-9,
    "max_bond_dim" : 8,
}

def test_small_circuit_final_density():
    assert np.linalg.norm(run_qa(CONFIG)[0] - run_qa_exact(CONFIG)[0]) < 1e-5
