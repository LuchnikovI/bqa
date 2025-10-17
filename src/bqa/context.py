from typing import Type, Callable
from functools import reduce
from typing import Iterable
from dataclasses import dataclass
from bqa.backends import Tensor

# types

Graph = list[list[int]]

Edge = tuple[int, int]

EdgeAmpl = tuple[Edge, float]

EdgeAmplIter = Iterable[tuple[Edge, float]]

EdgeToAmpl = dict[Edge, float]

NodeAmplsIter = Iterable[tuple[int, float]]

NodeToAmpl = dict[int, float]

EdgePosIter = Iterable[tuple[Edge, int]]

EdgeToPos = dict[Edge, int]

SubGraph = dict[int, list[int]]  # this is not fully valid subgraph since list[int] can point outside

DegreeToSubGraph = dict[int, SubGraph]

# graph abstraction

def _extend_list_by_defaults(lst: list, size: int, make_default_fn: Callable) -> None:
        while len(lst) < size:
                lst.append(make_default_fn())

def _make_default_node_content() -> list : return []

def _insert_edge(graph: Graph, edge_ampl: EdgeAmpl) -> Graph:
        (lhs_id, rhs_id), _ = edge_ampl
        min_required_nodes_number = max(lhs_id, rhs_id) + 1
        _extend_list_by_defaults(graph, min_required_nodes_number, _make_default_node_content)
        graph[lhs_id].append(rhs_id)
        graph[rhs_id].append(lhs_id)
        return graph

def _make_graph(edge_ampls: EdgeAmplIter) -> Graph:
        return reduce(_insert_edge, edge_ampls, [])

def _get_degree_to_subgraph(graph: Graph) -> DegreeToSubGraph:
        def insert_into_subgraph(
                        degree_to_subgraph: DegreeToSubGraph,
                        pair: tuple[int, list[int]]) -> DegreeToSubGraph:
                node_id, neighbors = pair
                degree = len(neighbors)
                degree_to_subgraph.setdefault(degree, {})[node_id] = neighbors
                return degree_to_subgraph
        return reduce(insert_into_subgraph, enumerate(graph), {})

# edge_to_ampl abstraction

def _make_edge_to_ampl_generator(edge_ampls: EdgeAmplIter) -> EdgeAmplIter:
        for (lhs_id, rhs_id), ampl in edge_ampls:
                yield (lhs_id, rhs_id), ampl
                yield (rhs_id, lhs_id), ampl

def _make_edge_to_ampl(edge_ampls: EdgeAmplIter) -> EdgeToAmpl:
        return dict(_make_edge_to_ampl_generator(edge_ampls))

# node_to_ampl abstraction

def _make_node_to_ampl(node_ampls: NodeAmplsIter) -> NodeToAmpl:
        return dict(node_ampls)

# edge_to_msg_pos abstraction

def _make_edge_to_msg_pos_generator(edge_ampls: EdgeAmplIter) -> EdgePosIter:
        counter = 0
        for (lhs_id, rhs_id), _ in edge_ampls:
                yield (lhs_id, rhs_id), counter
                yield (rhs_id, lhs_id), counter + 1
                counter = counter + 2

def _make_edge_to_msg_pos(edge_ampls: EdgeAmplIter) -> EdgeToPos:
        return dict(_make_edge_to_msg_pos_generator(edge_ampls))

# edge_to_lmbd_pos abstraction

def _make_edge_to_lmbd_pos_generator(edge_ampls: EdgeAmplIter) -> EdgePosIter:
        counter = 0
        for (lhs_id, rhs_id), _ in edge_ampls:
                yield (lhs_id, rhs_id), counter
                yield (rhs_id, lhs_id), counter
                counter = counter + 1

def _make_edge_to_lmbd_pos(edge_ampls: EdgeAmplIter) -> EdgeToPos:
        return dict(_make_edge_to_lmbd_pos_generator(edge_ampls))

# msg_pos_to_lmbd_pos abstraction

def _doubled_numbers_generator(number: int) -> Iterable[int]:
        for num in range(number // 2):
                yield num
                yield num

def _make_msgs_pos_to_lmbd_pos(edges_number: int, backend: Type[Tensor]) -> Tensor:
        return backend.from_iter(_doubled_numbers_generator(edges_number))

# layout abstraction

@dataclass
class Layout:
        node_ids: Tensor
        input_msgs_pos: list[Tensor]
        output_msgs_pos: list[Tensor]
        lmbds_pos: list[Tensor]

def _make_layout(subgraph: SubGraph,
                 edge_to_msg_pos: EdgeToPos,
                 edge_to_lmbd_pos: EdgeToPos,
                 backend: Type[Tensor]) -> Layout:

        def subgraph_record_to_input_msgs_pos(record: tuple[int, list[int]]) -> Iterable[int]:
                node_id, neighbors = record
                return map(lambda src: edge_to_msg_pos[(src, node_id)], neighbors)

        def subgraph_record_to_output_msgs_pos(record: tuple[int, list[int]]) -> Iterable[int]:
                node_id, neighbors = record
                return map(lambda dst: edge_to_msg_pos[(node_id, dst)], neighbors)

        def subgraph_record_to_lmbd_pos(record: tuple[int, list[int]]) -> Iterable[int]:
                node_id, neighbors = record
                return map(lambda src: edge_to_lmbd_pos[(src, node_id)], neighbors)

        node_ids = map(lambda pair: pair[0], subgraph.items())
        input_msgs_pos = map(subgraph_record_to_input_msgs_pos, subgraph.items())
        output_msgs_pos = map(subgraph_record_to_output_msgs_pos, subgraph.items())
        lmbds_pos = map(subgraph_record_to_lmbd_pos, subgraph.items())
        return Layout(backend.from_list(list(node_ids)),
                      list(map(backend.from_iter, zip(*input_msgs_pos))),
                      list(map(backend.from_iter, zip(*output_msgs_pos))),
                      list(map(backend.from_iter, zip(*lmbds_pos))))

def _make_degree_to_layout(graph: Graph,
                           edge_to_msg_pos: EdgeToPos,
                           edge_to_lmbd_pos: EdgeToPos,
                           backend: Type[Tensor]) -> dict[int, Layout]:
        return {d : _make_layout(sg, edge_to_msg_pos, edge_to_lmbd_pos, backend)
                for d, sg in _get_degree_to_subgraph(graph).items()}

# context abstraction

@dataclass
class Context:
        backend: Type[Tensor]
        graph: Graph
        edge_to_ampl: EdgeToAmpl
        node_to_ampl: NodeToAmpl
        edge_to_msg_pos: EdgeToPos
        edge_to_lmbd_pos: EdgeToPos
        msg_pos_to_lmbd_pos: Tensor
        degree_to_layout: dict[int, Layout]

        @property
        def qubits_number(self) -> int:
                return len(self.graph)

        @property
        def msgs_number(self) -> int:
                return len(self.edge_to_ampl)

        @property
        def lmbds_number(self) -> int:
                return self.msgs_number // 2

def build_context(spec: dict, backend: Type[Tensor]):

    def get_edges() -> EdgeAmplIter:
        for (lhs_id, rhs_id), ampl in filter(lambda key_val: isinstance(key_val[0], tuple), spec.items()):
                if (rhs_id, lhs_id) in spec:
                        raise ValueError(f"Edge connecting {lhs_id} and {rhs_id} is duplicated")
                yield (min(lhs_id, rhs_id), max(lhs_id, rhs_id)), ampl

    def get_nodes() -> NodeAmplsIter:
        return filter(lambda key_val: isinstance(key_val[0], int), spec.items())

    graph = _make_graph(get_edges())
    edge_to_ampl = _make_edge_to_ampl(get_edges())
    node_to_ampl = _make_node_to_ampl(get_nodes())
    edge_to_msg_pos = _make_edge_to_msg_pos(get_edges())
    edge_to_lmbd_pos = _make_edge_to_lmbd_pos(get_edges())
    msg_pos_to_lmbd_pos = _make_msgs_pos_to_lmbd_pos(len(edge_to_ampl), backend)
    degree_to_layout = _make_degree_to_layout(graph, edge_to_msg_pos, edge_to_lmbd_pos, backend)
    return Context(backend,
                   graph,
                   edge_to_ampl,
                   node_to_ampl,
                   edge_to_msg_pos,
                   edge_to_lmbd_pos,
                   msg_pos_to_lmbd_pos,
                   degree_to_layout)

