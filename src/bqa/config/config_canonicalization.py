from dataclasses import dataclass
from typing import Type
from numpy import array_equal, isclose
from bqa.backends import Tensor, BACKEND_STR_TO_BACKEND
from bqa.config.config_syntax import BACKEND_KEY, BP_EPS_KEY, DAMPING_KEY, DEFAULT_FIELD_KEY, EDGES_KEY, MAX_BOND_DIM_KEY, MAX_BP_ITER_NUMBER_KEY, MEASUREMENT_THRESHOLD_KEY, NODES_KEY, PINV_EPS_KEY, SCHEDULE_KEY, SEED_KEY
from bqa.config.schedule_canonicalization import _canonicalize_schedule
from bqa.config.utils import vectorized_append

# keywords

NODE_IDS_KEY = "node_ids"

NODE_AMPLS_KEY = "node_ampls"

EDGE_AMPLS_KEY = "edge_ampls"

LMBDS_POSITION_KEY = "lmbds_position"

INPUT_MSGS_POSITION_KEY = "input_msgs_position"

OUTPUT_MSGS_POSITION_KEY = "output_msgs_position"

# ------------

@dataclass
class Layout:
    node_ids: Tensor
    input_msgs_position: list[Tensor]
    output_msgs_position: list[Tensor]
    lmbds_position: list[Tensor]
    node_ampls: Tensor
    edge_ampls: list[Tensor]

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


def _make_degree_to_layout(
    graph,
    edge_to_msg_position,
    edge_to_lmbd_position,
    node_to_ampl,
    edge_to_ampl,
    backend,
):
    degree_to_layout = {}

    def get_layout_or_default(degree):
        layout = degree_to_layout.get(degree)
        if layout is None:
            degree_to_layout[degree] = {
                NODE_IDS_KEY : [],
                INPUT_MSGS_POSITION_KEY : [[] for _ in range(degree)],
                OUTPUT_MSGS_POSITION_KEY : [[] for _ in range(degree)],
                LMBDS_POSITION_KEY : [[] for _ in range(degree)],
                NODE_AMPLS_KEY : [],
                EDGE_AMPLS_KEY : [[] for _ in range(degree)],
            }
            return degree_to_layout[degree]
        else:
            return layout

    for node_id, nghbrng_ids in enumerate(graph):
        degree = len(nghbrng_ids)
        layout = get_layout_or_default(degree)
        layout[NODE_IDS_KEY].append(node_id)
        layout[NODE_AMPLS_KEY].append(node_to_ampl[node_id])
        vectorized_append(
            layout[INPUT_MSGS_POSITION_KEY],
            (edge_to_msg_position[(src_id, node_id)] for src_id in nghbrng_ids),
        )
        vectorized_append(
            layout[OUTPUT_MSGS_POSITION_KEY],
            (edge_to_msg_position[(node_id, dst_id)] for dst_id in nghbrng_ids),
        )
        vectorized_append(
            layout[LMBDS_POSITION_KEY],
            (edge_to_lmbd_position[(node_id, dst_id)] for dst_id in nghbrng_ids),
        )
        vectorized_append(
            layout[EDGE_AMPLS_KEY],
            (edge_to_ampl[(node_id, dst_id)] for dst_id in nghbrng_ids)
        )

    def dict_to_layout(dict_layout):
        return Layout(
            backend.make_from_list(dict_layout[NODE_IDS_KEY]),
            [
                backend.make_from_list(poss)
                for poss in dict_layout[INPUT_MSGS_POSITION_KEY]
            ],
            [
                backend.make_from_list(poss)
                for poss in dict_layout[OUTPUT_MSGS_POSITION_KEY]
            ],
            [backend.make_from_list(poss) for poss in dict_layout[LMBDS_POSITION_KEY]],
            backend.make_from_list(dict_layout[NODE_AMPLS_KEY]),
            [
                backend.make_from_list(ampl)
                for ampl in dict_layout[EDGE_AMPLS_KEY]
            ],
        )

    return {
        degree: dict_to_layout(dict_layout)
        for degree, dict_layout in degree_to_layout.items()
    }


def _make_paths_to_tensors(degree_to_layout):

    def make_paths_to_tensors_generator():
        for degree, layout in degree_to_layout.items():
            for pos, node_id in enumerate(layout.node_ids.numpy):
                yield (int(node_id), (degree, pos))

    return dict(make_paths_to_tensors_generator())


@dataclass
class Context:
    backend: Type[Tensor]
    graph: list[list[int]]
    bp_eps: float
    pinv_eps: float
    measurement_threshold: float
    nodes_number: int
    edges_number: int
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
        return self.edges_number // 2


def _get_nodes_number(config):
    return 1 + max(
        max(config[NODES_KEY].keys(), default=-1),
        max((max(lhs, rhs) for lhs, rhs in config[EDGES_KEY].keys())),
    )


def _get_edges_number(config):
    return len(config[EDGES_KEY])


def _make_graph(config, nodes_number):
    graph = [[] for _ in range(nodes_number)]
    for lhs, rhs in config[EDGES_KEY].keys():
        graph[lhs].append(rhs)
    return graph


def _make_edge_to_msg_position(config):
    return {edge: pos for pos, edge in enumerate(config[EDGES_KEY].keys())}


# this relies on order of edges in a dict
def _make_edge_to_lmbd_position(config):
    lmbds_number = len(config[EDGES_KEY]) // 2
    return {edge: pos % lmbds_number for pos, edge in enumerate(config[EDGES_KEY].keys())}


# this relies on order of edges in a dict
def _make_msg_position_to_lmbd_position(
    edges_number,
    backend,
):
    lmbds_number = edges_number // 2
    return backend.make_from_iter((pos % lmbds_number for pos in range(edges_number)))


def _canonicalize_config(config):
    nodes_number = _get_nodes_number(config)
    edges_number = _get_edges_number(config)
    backend = BACKEND_STR_TO_BACKEND[config[BACKEND_KEY]]
    graph = _make_graph(config, nodes_number)
    edge_to_msg_position = _make_edge_to_msg_position(config)
    edge_to_lmbd_position = _make_edge_to_lmbd_position(config)
    default_field = config[DEFAULT_FIELD_KEY]
    edge_to_ampl = config[EDGES_KEY]
    node_to_ampl = {node_id : config[NODES_KEY].get(node_id, default_field) for node_id in range(nodes_number)}
    msg_position_to_lmbd_position = _make_msg_position_to_lmbd_position(
        edges_number, backend
    )
    degree_to_layout = _make_degree_to_layout(
        graph,
        edge_to_msg_position,
        edge_to_lmbd_position,
        node_to_ampl,
        edge_to_ampl,
        backend,
    )
    return Context(
        backend,
        graph,
        config[BP_EPS_KEY],
        config[PINV_EPS_KEY],
        config[MEASUREMENT_THRESHOLD_KEY],
        nodes_number,
        edges_number,
        config[MAX_BOND_DIM_KEY],
        config[MAX_BP_ITER_NUMBER_KEY],
        config[SEED_KEY],
        msg_position_to_lmbd_position,
        degree_to_layout,
        _make_paths_to_tensors(degree_to_layout),
        list(_canonicalize_schedule(config[SCHEDULE_KEY])),
        edge_to_ampl,
        node_to_ampl,
        edge_to_msg_position,
        edge_to_lmbd_position,
        config[DAMPING_KEY]
    )

