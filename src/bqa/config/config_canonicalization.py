from dataclasses import dataclass
from functools import reduce
from itertools import islice
from typing import Type, TypeVar
from numpy import array_equal
from numpy.random import Generator, default_rng
from bqa.backends import NumPyBackend, Tensor
from bqa.config.config_syntax import Config, EdgeToAmpl, NodeToAmpl, Edge
from bqa.config.schedule_canonicalization import Instruction, _canonicalize_schedule

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

    def __eq__(
        self, value: object, /
    ) -> bool:  # needed only for testing, numpy conversion is ok
        if isinstance(value, Layout):
            return (
                array_equal(self.node_ids.numpy, value.node_ids.numpy)
                and len(self.input_msgs_position) == len(value.input_msgs_position)
                and all(
                    map(
                        lambda lhs, rhs: array_equal(lhs.numpy, rhs.numpy),
                        self.input_msgs_position,
                        value.input_msgs_position,
                    )
                )
                and len(self.output_msgs_position) == len(value.output_msgs_position)
                and all(
                    map(
                        lambda lhs, rhs: array_equal(lhs.numpy, rhs.numpy),
                        self.output_msgs_position,
                        value.output_msgs_position,
                    )
                )
                and len(self.lmbds_position) == len(value.lmbds_position)
                and all(
                    map(
                        lambda lhs, rhs: array_equal(lhs.numpy, rhs.numpy),
                        self.lmbds_position,
                        value.lmbds_position,
                    )
                )
            )
        else:
            return False


def _make_degree_to_layout(
    graph: Graph,
    edge_to_msg_position: EdgeToPosition,
    edge_to_lmbd_position: EdgeToPosition,
    backend: Type[Tensor],
) -> dict[int, Layout]:

    def get_input_msg_position(graph_record: tuple[int, list[int]]) -> list[int]:
        dst_id, src_ids = graph_record
        return list(map(lambda src_id: edge_to_msg_position[(src_id, dst_id)], src_ids))

    def get_output_msg_position(graph_record: tuple[int, list[int]]) -> list[int]:
        src_id, dst_ids = graph_record
        return list(map(lambda dst_id: edge_to_msg_position[(src_id, dst_id)], dst_ids))

    def get_lmbds_position(graph_record: tuple[int, list[int]]) -> list[int]:
        dst_id, src_ids = graph_record
        return list(
            map(lambda src_id: edge_to_lmbd_position[(dst_id, src_id)], src_ids)
        )

    def get_degree(graph_record: tuple[int, list[int]]) -> int:
        _, neighbor_ids = graph_record
        return len(neighbor_ids)

    def get_or_default(layouts: dict[int, DictLayout], degree: int) -> DictLayout:
        layout = layouts.get(degree)
        if layout is None:
            layouts[degree] = {
                "node_ids": [],
                "input_msgs_position": [],
                "output_msgs_position": [],
                "lmbds_position": [],
            }
            return layouts[degree]
        else:
            return layout

    def add_dict_layout_per_degree(
        dict_layouts: dict[int, DictLayout], graph_record: tuple[int, list[int]]
    ) -> dict[int, DictLayout]:
        degree = get_degree(graph_record)
        dict_layout = get_or_default(dict_layouts, degree)
        dict_layout["node_ids"].append(graph_record[0])
        dict_layout["input_msgs_position"].append(get_input_msg_position(graph_record))
        dict_layout["output_msgs_position"].append(
            get_output_msg_position(graph_record)
        )
        dict_layout["lmbds_position"].append(get_lmbds_position(graph_record))
        return dict_layouts

    def tensorize(list_of_positions: list[list[int]], degree: int) -> list[Tensor]:
        return list(
            map(
                lambda i: backend.from_iter(
                    map(lambda positions: positions[i], list_of_positions)
                ),
                range(degree),
            )
        )

    def dict_to_layout(dict_layout: DictLayout, degree: int) -> Layout:
        return Layout(
            backend.from_list(dict_layout["node_ids"]),
            tensorize(dict_layout["input_msgs_position"], degree),
            tensorize(dict_layout["output_msgs_position"], degree),
            tensorize(dict_layout["lmbds_position"], degree),
        )

    return {
        degree: dict_to_layout(dict_layout, degree)
        for degree, dict_layout in reduce(
            add_dict_layout_per_degree, enumerate(graph), {}
        ).items()
    }


@dataclass
class Context:
    backend: Type[Tensor]
    graph: Graph
    edge_to_ampl: EdgeToAmpl
    node_to_ampl: NodeToAmpl
    default_field: float
    bp_eps: float
    measurement_threshold: float
    nodes_number: int
    edges_number: int
    max_bond_dim: int
    max_bp_iters_number: int
    numpy_rng: Generator
    edge_to_msg_position: EdgeToPosition
    edge_to_lmbd_position: EdgeToPosition
    lmbd_aligned_ampls: Tensor
    msg_pos_to_lmbd_pos: Tensor
    degree_to_layout: dict[int, Layout]
    instructions: list[Instruction]

    @property
    def lmbds_number(self) -> int:
        return self.edges_number // 2


def _get_nodes_number(config: Config) -> int:
    return 1 + max(
        max(config["nodes"].keys(), default=-1),
        max(map(lambda edge: max(edge[0], edge[1]), config["edges"].keys())),
    )


def _get_edges_number(config: Config) -> int:
    return len(config["edges"])


def _make_graph(config: Config, nodes_number: int) -> Graph:
    def add_edge_to_graph(graph: Graph, edge: Edge) -> Graph:
        graph[edge[0]].append(edge[1])
        return graph

    graph = [[] for _ in range(nodes_number)]
    return reduce(add_edge_to_graph, config["edges"].keys(), graph)


def _make_edge_to_msg_position(config: Config) -> EdgeToPosition:
    return dict(map(lambda kv: (kv[1], kv[0]), enumerate(config["edges"].keys())))


# this relies on order of edges in a dict
def _make_edge_to_lmbd_position(config: Config) -> EdgeToPosition:
    lmbds_number = len(config["edges"]) // 2
    return dict(map(lambda kv: (kv[1], kv[0] % lmbds_number), enumerate(config["edges"].keys())))

# this relies on order of edges in a dict
def _make_msg_position_to_lmbd_position(
    edges_number: int,
    backend: Type[Tensor],
) -> Tensor:
    lmbds_number = edges_number // 2
    return backend.from_iter(map(lambda x: x % lmbds_number, range(edges_number)))

# this relies on order of edges in a dict
def _make_lmbd_aligned_amplitudes(
        edge_to_ampl: EdgeToAmpl,
        backend: Type[Tensor],
) -> Tensor:
    lmbds_number = len(edge_to_ampl) // 2
    return backend.from_iter(islice(edge_to_ampl.values(), lmbds_number))

def _canonicalize_config(config: Config) -> Context:
    nodes_number = _get_nodes_number(config)
    edges_number = _get_edges_number(config)
    backend = BACKEND_STR_TO_BACKEND[config["backend"]]
    graph = _make_graph(config, nodes_number)
    edge_to_msg_position = _make_edge_to_msg_position(config)
    edge_to_lmbd_position = _make_edge_to_lmbd_position(config)
    msg_position_to_lmbd_position = _make_msg_position_to_lmbd_position(
        edges_number, backend
    )
    return Context(
        backend,
        graph,
        config["edges"],
        config["nodes"],
        config["default_field"],
        config["bp_eps"],
        config["measurement_threshold"],
        nodes_number,
        edges_number,
        config["max_bond_dim"],
        config["max_bp_iters_number"],
        default_rng(config["seed"]),
        edge_to_msg_position,
        edge_to_lmbd_position,
        _make_lmbd_aligned_amplitudes(config["edges"], backend),
        msg_position_to_lmbd_position,
        _make_degree_to_layout(
            graph, edge_to_msg_position, edge_to_lmbd_position, backend
        ),
        list(_canonicalize_schedule(config["schedule"])),
    )
