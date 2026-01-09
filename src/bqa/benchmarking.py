import logging
from random import Random
from typing import Any, Callable

from networkx import Graph, nx_agraph

try:
    from networkx import random_regular_graph, grid_2d_graph, convert_node_labels_to_integers
except ImportError as _:
    logging.warning("NextworkX is not found, some benchmarking utils are unavailable, to install NetworkX run `pip install networkx`")

    def random_regular_graph(*args, **kwargs) -> Any:
        raise NotImplementedError("NetworkX is not found")

    def grid_2d_graph(*args, **kwargs) -> Any:
        raise NotImplementedError("NetworkX is not found")

    def convert_node_labels_to_integers(*args, **kwargs) -> Any:
        raise NotImplementedError("NetworkX is not found")


log = logging.getLogger(__name__)

Edge = tuple[int, int]

EdgeToAmpl = dict[Edge, float]

NodeToAmpl = dict[int, float]


def _default_node_ampl_func(rng: Random, _: int) -> float:
    return rng.uniform(-1.0, 1.0)


def _default_edge_ampl_func(rng: Random, _: tuple[int, int]) -> float:
    return rng.uniform(-1.0, 1.0)


def _get_from_nx_graph(
        nxgraph: Graph,
        rng: Random,
        node_ampl_func: Callable[[Random, int], float],
        edge_ampl_func: Callable[[Random, Edge], float],
) -> tuple[NodeToAmpl, EdgeToAmpl]:
    nxgraph_enum = convert_node_labels_to_integers(nxgraph)
    nodes = {node_id: node_ampl_func(rng, node_id) for node_id in nxgraph_enum.nodes}
    edges = {edge: edge_ampl_func(rng, edge) for edge in nxgraph_enum.edges}
    return nodes, edges


def generate_qubo_on_random_regular_graph(
        nodes_number: int,
        degree: int = 3,
        seed: int = 42,
        node_ampl_func: Callable[[Random, int], float] = _default_node_ampl_func,
        edge_ampl_func: Callable[[Random, Edge], float] = _default_edge_ampl_func,
) -> tuple[NodeToAmpl, EdgeToAmpl]:
    rng = Random(seed)
    nxgraph = random_regular_graph(degree, nodes_number, rng)
    return _get_from_nx_graph(nxgraph, rng, node_ampl_func, edge_ampl_func)


def generate_qubo_on_2d_grid(
        m: int,
        n: int,
        seed: int = 42,
        node_ampl_func: Callable[[Random, int], float] = _default_node_ampl_func,
        edge_ampl_func: Callable[[Random, Edge], float] = _default_edge_ampl_func,
) -> tuple[NodeToAmpl, EdgeToAmpl]:
    rng = Random(seed)
    nxgraph = grid_2d_graph(m, n)
    return _get_from_nx_graph(nxgraph, rng, node_ampl_func, edge_ampl_func)

