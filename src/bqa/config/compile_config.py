from dataclasses import dataclass
from typing import Type
from numpy import array_equal, isclose
from bqa.backends import BACKEND_STR_TO_BACKEND, Tensor
from bqa.config.desugar_config import canonicalize_edge_id
from bqa.config.metrics import get_edges_number, get_nodes_number
from bqa.config.validate_config import (ACTIONS_KEY, BACKEND_KEY, BP_EPS_KEY, DAMPING_KEY, EDGES_KEY, DEFAULT_FIELD_KEY,
                                        FINAL_MIXING_KEY, INITIAL_MIXING_KEY,  MAX_BOND_DIM_KEY, MAX_BP_ITER_NUMBER_KEY, MEASUREMENT_THRESHOLD_KEY, NODES_KEY,
                                        PINV_EPS_KEY, SCHEDULE_KEY, SEED_KEY, STEPS_NUMBER_KEY, TOTAL_TIME_KEY, WEIGHT_KEY)
from bqa.config.pipeline import pipeline

@dataclass
class RawLayout:
    node_ids: list[int]
    input_msgs_position: list[list[int]]
    output_msgs_position: list[list[int]]
    lmbds_position: list[list[int]]
    node_ampls: list[float]
    edge_ampls: list[list[float]]

    @classmethod
    def new_empty(cls, degree):
        return cls(
            [],
            [[] for _ in range(degree)],
            [[] for _ in range(degree)],
            [[] for _ in range(degree)],
            [],
            [[] for _ in range(degree)],
        )


@dataclass
class Layout:
    node_ids: Tensor
    input_msgs_position: list[Tensor]
    output_msgs_position: list[Tensor]
    lmbds_position: list[Tensor]
    node_ampls: Tensor
    edge_ampls: list[Tensor]

    @classmethod
    def from_raw(cls, backend, raw_layout):
        return cls(
            backend.make_from_list(raw_layout.node_ids),
            list(map(backend.make_from_list, raw_layout.input_msgs_position)),
            list(map(backend.make_from_list, raw_layout.output_msgs_position)),
            list(map(backend.make_from_list, raw_layout.lmbds_position)),
            backend.make_from_list(raw_layout.node_ampls),
            list(map(backend.make_from_list, raw_layout.edge_ampls)),
        )

    def __eq__(self, value):  # needed only for testing, thus conversion to numpy is ok

        def is_equal_tensors(lhss, rhss):
            return all(
                map(lambda lhs, rhs: array_equal(lhs.numpy, rhs.numpy), lhss, rhss)
            )

        def is_close_tensors(lhss, rhss):
            return all(
                map(lambda lhs, rhs: isclose(lhs.numpy, rhs.numpy).all(), lhss, rhss)
            )

        if isinstance(value, Layout):
            return (
                len(self.input_msgs_position) == len(value.input_msgs_position)
                and len(self.output_msgs_position) == len(value.output_msgs_position)
                and len(self.lmbds_position) == len(value.lmbds_position)
                and array_equal(self.node_ids.numpy, value.node_ids.numpy)
                and is_equal_tensors(self.input_msgs_position, value.input_msgs_position)
                and is_equal_tensors(self.output_msgs_position, value.output_msgs_position)
                and is_equal_tensors(self.lmbds_position, value.lmbds_position)
                and bool(isclose(self.node_ampls.numpy, value.node_ampls.numpy).all())
                and is_close_tensors(self.edge_ampls, value.edge_ampls)
            )
        else:
            return False


@dataclass
class Context:
    backend: Type[Tensor]
    graph: list[list[int]]
    bp_eps: float
    pinv_eps: float
    measurement_threshold: float
    nodes_number: int
    msgs_number: int
    max_bond_dim: int
    max_bp_iters_number: int
    seed: int
    msg_pos_to_lmbd_pos: Tensor
    degree_to_layout: dict[int, Layout]
    path_to_tensors: dict[int, tuple[int, int]]
    instructions: list
    edge_to_ampl: dict[tuple[int, int], float]
    node_to_ampl: dict[int, float]
    edge_to_msg_pos: dict[tuple[int, int], int]
    edge_to_lmbd_pos: dict[tuple[int, int], int]
    damping: float

    @property
    def lmbds_number(self):
        return self.msgs_number // 2


def make_graph(config, nodes_number):
    graph = [[] for _ in range(nodes_number)]
    for lhs, rhs in config[EDGES_KEY].keys():
        graph[lhs].append(rhs)
        graph[rhs].append(lhs)
    return graph


def directed_edges_iter(config):
    for edge_id in config[EDGES_KEY].keys():
        yield edge_id
    for lhs_id, rhs_id in config[EDGES_KEY].keys():
        yield rhs_id, lhs_id


def make_edge_to_msg_position(config):
    return {edge: pos for pos, edge in enumerate(directed_edges_iter(config))}


def make_edge_to_lmbd_position(config):
    lmbds_number = get_edges_number(config)
    return {edge: pos % lmbds_number for pos, edge in enumerate(directed_edges_iter(config))}

def make_edge_to_amplitue(config):
    edges = config[EDGES_KEY]
    return {k : edges[canonicalize_edge_id(k)] for k in directed_edges_iter(config)}


def make_msg_position_to_lmbd_position(
    lmbds_number,
    backend,
):
    return backend.make_from_iter((pos % lmbds_number for pos in range(2 * lmbds_number)))


def make_degree_to_layout(
    graph,
    edge_to_msg_position,
    edge_to_lmbd_position,
    node_to_ampl,
    edge_to_ampl,
    backend,
):
    degree_to_layout = {}
    for node_id, neighs_ids in enumerate(graph):
        degree = len(neighs_ids)
        layout = degree_to_layout.get(degree)
        if layout is None:
            degree_to_layout[degree] = RawLayout.new_empty(degree)
            layout = degree_to_layout[degree]
        layout.node_ids.append(node_id)
        layout.node_ampls.append(node_to_ampl[node_id])
        for slot, neigh_id in enumerate(neighs_ids):
            layout.input_msgs_position[slot].append(edge_to_msg_position[(neigh_id, node_id)])
            layout.output_msgs_position[slot].append(edge_to_msg_position[(node_id, neigh_id)])
            layout.lmbds_position[slot].append(edge_to_lmbd_position[(node_id, neigh_id)])
            layout.edge_ampls[slot].append(edge_to_ampl[(node_id, neigh_id)])
    return {
        degree: Layout.from_raw(backend, dict_layout)
        for degree, dict_layout in degree_to_layout.items()
    }


def make_paths_to_tensors_generator(degree_to_layout):
    for degree, layout in degree_to_layout.items():
        for pos, node_id in enumerate(layout.node_ids.numpy):
            yield (int(node_id), (degree, pos))


def make_linearly_interpolation_generator(start, end, steps_number):
    delta = (end - start) / steps_number
    for n in range(1, steps_number + 1):
        yield start + n * delta


def make_complex_action_instr_generator(total_time, action):
    final_mixing = action[FINAL_MIXING_KEY]
    initial_mixing = action[INITIAL_MIXING_KEY]
    steps_number = action[STEPS_NUMBER_KEY]
    time_step = total_time * action[WEIGHT_KEY] / steps_number
    for p in make_linearly_interpolation_generator(initial_mixing, final_mixing, steps_number):
        yield {"xtime": p * time_step, "ztime": (1.0 - p) * time_step}


def make_compile_schedule_generator(schedule):
    total_time = schedule[TOTAL_TIME_KEY]
    actions = schedule[ACTIONS_KEY]
    for action in actions:
        if isinstance(action, str):
            yield action
        elif isinstance(action, dict):
            yield from make_complex_action_instr_generator(total_time, action)
        else:
            assert False

@pipeline
def compile_config(config):
    nodes_number = get_nodes_number(config)
    edges_number = get_edges_number(config)
    backend = BACKEND_STR_TO_BACKEND[config[BACKEND_KEY]]
    graph = make_graph(config, nodes_number)
    edge_to_msg_position = make_edge_to_msg_position(config)
    edge_to_lmbd_position = make_edge_to_lmbd_position(config)
    default_field = config[DEFAULT_FIELD_KEY]
    edge_to_ampl = make_edge_to_amplitue(config)
    node_to_ampl = {node_id : config[NODES_KEY].get(node_id, default_field) for node_id in range(nodes_number)}
    msg_position_to_lmbd_position = make_msg_position_to_lmbd_position(
        edges_number, backend
    )
    degree_to_layout = make_degree_to_layout(
        graph,
        edge_to_msg_position,
        edge_to_lmbd_position,
        node_to_ampl,
        edge_to_ampl,
        backend,
    )
    return Context(
        backend = backend,
        graph = graph,
        bp_eps = config[BP_EPS_KEY],
        pinv_eps = config[PINV_EPS_KEY],
        measurement_threshold = config[MEASUREMENT_THRESHOLD_KEY],
        nodes_number = nodes_number,
        msgs_number = 2 * edges_number,
        max_bond_dim = config[MAX_BOND_DIM_KEY],
        max_bp_iters_number = config[MAX_BP_ITER_NUMBER_KEY],
        seed = config[SEED_KEY],
        msg_pos_to_lmbd_pos = msg_position_to_lmbd_position,
        degree_to_layout = degree_to_layout,
        path_to_tensors = dict(make_paths_to_tensors_generator(degree_to_layout)),
        instructions = list(make_compile_schedule_generator(config[SCHEDULE_KEY])),
        edge_to_ampl = edge_to_ampl,
        node_to_ampl = node_to_ampl,
        edge_to_msg_pos = edge_to_msg_position,
        edge_to_lmbd_pos = edge_to_lmbd_position,
        damping = config[DAMPING_KEY],
    )

