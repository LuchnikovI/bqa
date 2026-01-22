from itertools import chain
import logging
from math import isclose
from operator import sub
import numpy as np
from bqa.backends import NumPyBackend
from bqa.config.core import config_to_context
from bqa.config.config_canonicalization import Layout

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def assert_isclose_numeric_dicts_order_sensetive(lhs: dict, rhs: dict) -> None:
    assert len(lhs) == len(rhs)
    for (lhs_key, lhs_val), (rhs_key, rhs_val) in zip(lhs.items(), rhs.items()):
        assert lhs_key == rhs_key
        assert isclose(lhs_val, rhs_val)


def sum_x_and_y_times(instruction) -> float:
    assert isinstance(instruction, dict)
    return instruction["xtime"] + instruction["ztime"]


def div_x_by_sum_x_and_y_times(instruction) -> float:
    assert isinstance(instruction, dict)
    xtime = instruction["xtime"]
    return xtime / (instruction["ztime"] + xtime)


def test_config_to_context():
    context = config_to_context(
        {
            "edges": {
                (0, 2): 1.0,
                (1, 0): -1,
                (3, 0): 0.1,
                (2, 4): 1.1,
                (1, 4): 0,
                (3, 1): 1,
            },
            "nodes": {2: 1, 6: -1.1},
            "default_field": -0.5,
            "schedule": {
                "starting_mixing": 0.8,
                "total_time": 5,
                "actions": [
                    {"weight": 0.4, "final_mixing": 0.3, "steps_number": 8},
                    "measure",
                    {"type": "imag_time_evolution", "weight": 0.6, "final_mixing": 0.11, "steps_number": 10},
                    "get_bloch_vectors",
                ],
            },
            "damping" : 0.3,
        },
    )
    assert context.edges_number == 12
    assert context.nodes_number == 7
    assert context.max_bp_iters_number == 100
    assert isclose(context.bp_eps, 1e-10)
    assert isclose(context.damping, 0.3)
    assert context.max_bond_dim == 4
    assert len(context.degree_to_layout) == 3
    assert context.degree_to_layout[0] == Layout(
        NumPyBackend(np.array([5, 6], dtype=np.intp)),
        [],
        [],
        [],
        NumPyBackend(np.array([-0.5, -1.1])),
        [],
    )
    assert context.degree_to_layout[2] == Layout(
        NumPyBackend(np.array([2, 3, 4], dtype=np.intp)),
        [
            NumPyBackend(np.array([9, 8, 3], dtype=np.intp)),
            NumPyBackend(np.array([0, 11, 4], dtype=np.intp)),
        ],
        [
            NumPyBackend(np.array([3, 2, 9], dtype=np.intp)),
            NumPyBackend(np.array([6, 5, 10], dtype=np.intp)),
        ],
        [
            NumPyBackend(np.array([3, 2, 3], dtype=np.intp)),
            NumPyBackend(np.array([0, 5, 4], dtype=np.intp)),
        ],
        NumPyBackend(np.array([1., -0.5, -0.5])),
        [
            NumPyBackend(np.array([1.1, 0.1, 1.1])),
            NumPyBackend(np.array([1., 1., 0.])),
        ]
    )
    assert context.degree_to_layout[3] == Layout(
        NumPyBackend(np.array([0, 1], dtype=np.intp)),
        [
            NumPyBackend(np.array([6, 7], dtype=np.intp)),
            NumPyBackend(np.array([1, 10], dtype=np.intp)),
            NumPyBackend(np.array([2, 5], dtype=np.intp)),
        ],
        [
            NumPyBackend(np.array([0, 1], dtype=np.intp)),
            NumPyBackend(np.array([7, 4], dtype=np.intp)),
            NumPyBackend(np.array([8, 11], dtype=np.intp)),
        ],
        [
            NumPyBackend(np.array([0, 1], dtype=np.intp)),
            NumPyBackend(np.array([1, 4], dtype=np.intp)),
            NumPyBackend(np.array([2, 5], dtype=np.intp)),
        ],
        NumPyBackend(np.array([-0.5, -0.5])),
        [
            NumPyBackend(np.array([1., -1.])),
            NumPyBackend(np.array([-1., 0.])),
            NumPyBackend(np.array([0.1, 1.])),
        ],
    )
    total_time = sum(
        map(
            sum_x_and_y_times,
            filter(lambda x: isinstance(x, dict), context.instructions),
        )
    )
    assert isclose(total_time, 5.0)
    first_part_time = sum(
        map(
            sum_x_and_y_times,
            filter(lambda x: isinstance(x, dict), context.instructions[:8]),
        )
    )
    second_part_time = sum(
        map(
            sum_x_and_y_times,
            filter(lambda x: isinstance(x, dict), context.instructions[8:]),
        )
    )
    assert isclose(first_part_time, 5.0 * 0.4)
    assert isclose(second_part_time, 5.0 * 0.6)
    mixings = list(
        map(
            div_x_by_sum_x_and_y_times,
            filter(lambda x: isinstance(x, dict), context.instructions),
        )
    )
    first_part_mixings = mixings[:8]
    second_part_mixings = mixings[8:]
    assert isclose(first_part_mixings[0], 0.8)
    assert isclose(second_part_mixings[0], 0.3)
    first_part_diffs = list(
        map(
            sub,
            chain(first_part_mixings[1:], [second_part_mixings[0]]),
            first_part_mixings,
        )
    )
    second_part_diffs = list(
        map(sub, chain(second_part_mixings[1:], [0.11]), second_part_mixings)
    )
    assert all(map(lambda x: isclose(x, first_part_diffs[0]), first_part_diffs))
    assert all(map(lambda x: isclose(x, second_part_diffs[0]), second_part_diffs))
    for node_id, (degree, pos) in context.path_to_tensors.items():
        assert context.degree_to_layout[degree].node_ids.numpy[pos] == node_id
    print("Test `config_to_context`: OK")

