from dataclasses import dataclass
from typing import Iterable, Iterator, Type, TypeVar
from numpy import array_equal, isclose
from bqa.backends import NumPyBackend, Tensor
from bqa.config.config_syntax import Config, Edge, EdgeToAmpl, NodeToAmpl
from bqa.config.schedule_canonicalization import Instruction, _canonicalize_schedule
from bqa.config.utils import vectorized_append

T1 = TypeVar("T1")

T2 = TypeVar("T2")

Graph = list[list[int]]

DictLayout = dict  # dict[str, list[list[int]] | list[int]]

EdgeToPosition = dict[Edge, int]

BACKEND_STR_TO_BACKEND = {"numpy": NumPyBackend}


@dataclass
class Layout:
    node_ids: Tensor
    input_msgs_position: list[Tensor]
    output_msgs_position: list[Tensor]
    lmbds_position: list[Tensor]
    node_ampls: Tensor
    edge_ampls: list[Tensor]

    def __eq__(
        self, value: object, /
    ) -> bool:  # needed only for testing, numpy conversion is ok

        def is_equal_tensors(lhss: Iterable[Tensor], rhss: Iterable[Tensor]) -> bool:
            return all(
                map(lambda lhs, rhs: array_equal(lhs.numpy, rhs.numpy), lhss, rhss)
            )

        def is_close_tensors(lhss: Iterable[Tensor], rhss: Iterable[Tensor]) -> bool:
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
    graph: Graph,
    edge_to_msg_position: EdgeToPosition,
    edge_to_lmbd_position: EdgeToPosition,
    node_to_ampl: NodeToAmpl,
    edge_to_ampl: EdgeToAmpl,
    backend: Type[Tensor],
) -> dict[int, Layout]:
    degree_to_layout = {}

    def get_layout_or_default(degree: int) -> DictLayout:
        layout = degree_to_layout.get(degree)
        if layout is None:
            degree_to_layout[degree] = {
                "node_ids": [],
                "input_msgs_position": [[] for _ in range(degree)],
                "output_msgs_position": [[] for _ in range(degree)],
                "lmbds_position": [[] for _ in range(degree)],
                "node_ampls" : [],
                "edge_ampls" : [[] for _ in range(degree)],
            }
            return degree_to_layout[degree]
        else:
            return layout

    for node_id, nghbrng_ids in enumerate(graph):
        degree = len(nghbrng_ids)
        layout = get_layout_or_default(degree)
        layout["node_ids"].append(node_id)
        layout["node_ampls"].append(node_to_ampl[node_id])
        vectorized_append(
            layout["input_msgs_position"],
            (edge_to_msg_position[(src_id, node_id)] for src_id in nghbrng_ids),
        )
        vectorized_append(
            layout["output_msgs_position"],
            (edge_to_msg_position[(node_id, dst_id)] for dst_id in nghbrng_ids),
        )
        vectorized_append(
            layout["lmbds_position"],
            (edge_to_lmbd_position[(node_id, dst_id)] for dst_id in nghbrng_ids),
        )
        vectorized_append(
            layout["edge_ampls"],
            (edge_to_ampl[(node_id, dst_id)] for dst_id in nghbrng_ids)
        )

    def dict_to_layout(dict_layout: DictLayout) -> Layout:
        return Layout(
            backend.make_from_list(dict_layout["node_ids"]),
            [
                backend.make_from_list(poss)
                for poss in dict_layout["input_msgs_position"]
            ],
            [
                backend.make_from_list(poss)
                for poss in dict_layout["output_msgs_position"]
            ],
            [backend.make_from_list(poss) for poss in dict_layout["lmbds_position"]],
            backend.make_from_list(dict_layout["node_ampls"]),
            [
                backend.make_from_list(ampl)
                for ampl in dict_layout["edge_ampls"]
            ],
        )

    return {
        degree: dict_to_layout(dict_layout)
        for degree, dict_layout in degree_to_layout.items()
    }


def _make_paths_to_tensors(
    degree_to_layout: dict[int, Layout],
) -> dict[int, tuple[int, int]]:
    def make_generator_paths_to_tensors() -> Iterator[tuple[int, tuple[int, int]]]:
        for degree, layout in degree_to_layout.items():
            for pos, node_id in enumerate(layout.node_ids.numpy):
                yield (int(node_id), (degree, pos))

    return dict(make_generator_paths_to_tensors())


@dataclass
class Context:
    backend: Type[Tensor]
    graph: Graph
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
    instructions: list[Instruction]
    edge_to_ampl: EdgeToAmpl
    node_to_ampl: NodeToAmpl
    edge_to_msg_pos: EdgeToPosition
    edge_to_lmbd_pos: EdgeToPosition

    @property
    def lmbds_number(self) -> int:
        return self.edges_number // 2


def _get_nodes_number(config: Config) -> int:
    return 1 + max(
        max(config["nodes"].keys(), default=-1),
        max((max(lhs, rhs) for lhs, rhs in config["edges"].keys())),
    )


def _get_edges_number(config: Config) -> int:
    return len(config["edges"])


def _make_graph(config: Config, nodes_number: int) -> Graph:
    graph = [[] for _ in range(nodes_number)]
    for lhs, rhs in config["edges"].keys():
        graph[lhs].append(rhs)
    return graph


def _make_edge_to_msg_position(config: Config) -> EdgeToPosition:
    return {edge: pos for pos, edge in enumerate(config["edges"].keys())}


# this relies on order of edges in a dict
def _make_edge_to_lmbd_position(config: Config) -> EdgeToPosition:
    lmbds_number = len(config["edges"]) // 2
    return {edge: pos % lmbds_number for pos, edge in enumerate(config["edges"].keys())}


# this relies on order of edges in a dict
def _make_msg_position_to_lmbd_position(
    edges_number: int,
    backend: Type[Tensor],
) -> Tensor:
    lmbds_number = edges_number // 2
    return backend.make_from_iter((pos % lmbds_number for pos in range(edges_number)))


def _canonicalize_config(config: Config) -> Context:
    nodes_number = _get_nodes_number(config)
    edges_number = _get_edges_number(config)
    backend = BACKEND_STR_TO_BACKEND[config["backend"]]
    graph = _make_graph(config, nodes_number)
    edge_to_msg_position = _make_edge_to_msg_position(config)
    edge_to_lmbd_position = _make_edge_to_lmbd_position(config)
    default_field = config["default_field"]
    edge_to_ampl = config["edges"]
    node_to_ampl = {node_id : config["nodes"].get(node_id, default_field) for node_id in range(nodes_number)}
    config["nodes"]
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
        config["bp_eps"],
        config["pinv_eps"],
        config["measurement_threshold"],
        nodes_number,
        edges_number,
        config["max_bond_dim"],
        config["max_bp_iters_number"],
        config["seed"],
        msg_position_to_lmbd_position,
        degree_to_layout,
        _make_paths_to_tensors(degree_to_layout),
        list(_canonicalize_schedule(config["schedule"])),
        edge_to_ampl,
        node_to_ampl,
        edge_to_msg_position,
        edge_to_lmbd_position,
    )
