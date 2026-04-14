import logging
from heapq import heapify, heappop, heappush
from bqa.config.schedule_syntax import GET_BLOCH_VECTORS
from .config_syntax import (
    DEFAULT_NODES, EDGES_KEY, NODES_KEY, DEFAULT_FIELD_KEY, DEFAULT_DEFAULT_FIELD,
    ConfigSyntaxError, _get_or_default_and_warn, _get_or_raise, _analyse_edge_id,
    _analyse_number, _analyse_non_neg_int)

log = logging.getLogger(__name__)

SPARSIFICATION_KEY = "sparsification"

CLUSTER_COUPLING_KEY = "cluster_coupling"

MAXIMAL_DEGREE_KEY = "maximal_degree"

BLOCH_VECTORS_KEY = "bloch_vectors"

MEASUREMENT_OUTCOMES_KEY = "measurement_outcomes"

DEFAULT_COUPLING_AMPLITUDE = 1.0

DEFAULT_MAXIMAL_DEGREE = 3

def _analyse_maximal_degree(maximal_degree):
    if isinstance(maximal_degree, int) and maximal_degree >= 3:
        return maximal_degree
    else:
        raise ConfigSyntaxError(f"Invalid maximal degree {maximal_degree}, must be an `int` value >= 3")


def get_iter(pairs, name):
    if isinstance(pairs, (list, tuple)):
        return pairs
    elif isinstance(pairs, dict):
        return pairs.items()
    else:
        raise ConfigSyntaxError(f"{name} must be either dict, list or a tuple, but got {type(pairs)}")


def canonicalize_edge_id(edge_id):
    i, j = edge_id
    assert i != j
    return (i, j) if i < j else (j, i)


def gen_validated_edges(edges):
    edges_iter = get_iter(edges, "Edges")
    seen = set()
    for edge in edges_iter:
        if isinstance(edge, (tuple, list)) and len(edge) == 2:
            try:
                edge_id = canonicalize_edge_id(_analyse_edge_id(edge[0]))
                coupling = _analyse_number(edge[1])
                if edge_id in seen:
                    raise ConfigSyntaxError(f"Repeated edge {edge_id}")
                seen.add(edge_id)
                yield edge_id, coupling
            except ConfigSyntaxError as e:
                raise ConfigSyntaxError(f"Invalid edge {edge}") from e
        else:
            raise ConfigSyntaxError(f"Edge must be a list or tuple with two elements, but recived {edge}")


def gen_validated_nodes(nodes):
    nodes_iter = get_iter(nodes, "Nodes")
    seen = set()
    for node in nodes_iter:
        if isinstance(node, (tuple, list)) and len(node) == 2:
            try:
                node_id = _analyse_non_neg_int(node[0])
                field = _analyse_number(node[1])
                if node_id in seen:
                    raise ConfigSyntaxError(f"Repeated node {node_id}")
                seen.add(node_id)
                yield node_id, field
            except ConfigSyntaxError as e:
                raise ConfigSyntaxError(f"Invalid node {node}") from e
        else:
            raise ConfigSyntaxError(f"Node must be a list or tuple with two elements, but recieved {node}")


def make_bins_alignment(max_degree, total_degree):
    assert total_degree > max_degree
    out_degree = max_degree - 1
    in_degree = max_degree - 2
    bins = [[0, -out_degree, 0]]
    ptr = out_degree
    bin_id = 1
    while total_degree - ptr > out_degree:
        bins.append([0, -in_degree, bin_id])
        ptr += in_degree
        bin_id += 1
    bins.append([0, ptr - total_degree, bin_id])
    heapify(bins)
    return bins

class Problem:
    def __init__(self, edges, nodes, default_field):
        try:
            self.default_field = default_field
            self.nodes = dict(gen_validated_nodes(nodes))
            self.edges = dict(gen_validated_edges(edges))
            self.graph_size = max(
                max(*self.nodes.keys()),
                max(*(max(i, j) for i, j in self.edges.keys()))
            ) + 1
            self.graph = [set() for _ in range(self.graph_size)]
            for i, j in self.edges.keys():
                self.graph[i].add(j)
                self.graph[j].add(i)
        except ConfigSyntaxError as e:
            raise ConfigSyntaxError("Invalid problem specification") from e

    def get_field(self, node_id):
        return self.nodes.get(node_id, self.default_field) if node_id < self.graph_size else None

    def get_degree(self, node_id):
        assert self.graph_size > node_id
        return len(self.graph[node_id])

    def add_node(self, field):
        node_id = self.graph_size
        self.graph.append(set())
        self.nodes[node_id] = field
        self.graph_size += 1
        return node_id

    def add_edge(self, i, j, coupling):
        edge_id = canonicalize_edge_id((i, j))
        assert edge_id not in self.edges
        self.edges[edge_id] = coupling
        self.graph[i].add(j)
        self.graph[j].add(i)

    def remove_edge(self, i, j):
        edge_id = canonicalize_edge_id((i, j))
        assert edge_id in self.edges
        coupling = self.edges[edge_id]
        del self.edges[edge_id]
        self.graph[i].remove(j)
        self.graph[j].remove(i)
        return coupling

    def move_edge(self, i, old_j, new_j):
        assert i != old_j
        assert i != new_j
        assert i < self.graph_size
        assert old_j < self.graph_size
        assert new_j < self.graph_size
        coupling = self.remove_edge(i, old_j)
        self.add_edge(i, new_j, coupling)

    def split_neighbors(self, node_id, max_degree):
        neighs = self.graph[node_id]
        total_degree = len(neighs)
        bins = make_bins_alignment(max_degree, total_degree)
        bins_number = len(bins)
        coupling_and_id = [(abs(self.edges[canonicalize_edge_id((neigh_id, node_id))]), neigh_id) for neigh_id in neighs]
        coupling_and_id.sort(key = lambda x: x[0], reverse=True)
        cluster = [[] for _ in range(bins_number)]
        for coupling, neigh_id in coupling_and_id:
            while True:
                assert bins
                load, neg_cap, bin_id = heappop(bins)
                if neg_cap < 0:
                    break
            cluster[bin_id].append(neigh_id)
            heappush(bins, [load + coupling, neg_cap + 1, bin_id])
        bins.sort(key = lambda x: x[2])
        total_load = sum(map(lambda x: x[0], coupling_and_id))
        return cluster, total_load

    def sparsify_problem(self, max_degree, ampl):
        info = {"size": self.graph_size, "node_to_childs" : {}}
        node_to_childs = info["node_to_childs"]
        for node_id in range(self.graph_size):
            if self.get_degree(node_id) > max_degree:
                last_chain_node_id = node_id
                cluster, total_load = self.split_neighbors(node_id, max_degree)
                cluster_size = len(cluster)
                field = self.get_field(node_id)
                assert field is not None
                cluster_coupling = total_load + abs(field)
                new_field = field / cluster_size
                self.nodes[node_id] = new_field
                node_to_childs[node_id] = []
                for neighs in cluster[1:]:
                    new_node_id = self.add_node(new_field)
                    node_to_childs[node_id].append(new_node_id)
                    for neigh_id in neighs:
                        self.move_edge(neigh_id, node_id, new_node_id)
                    self.add_edge(last_chain_node_id, new_node_id, -ampl * cluster_coupling)
                    last_chain_node_id = new_node_id
        return info

    def release(self):
        return {
            "nodes": self.nodes,
            "edges": self.edges,
        }


def preprocess(
        config,
):
    if isinstance(config, dict):
        sparsification = config.get(SPARSIFICATION_KEY)
        if sparsification is None:
            return config, None
        if isinstance(sparsification, dict):
            ampl = _analyse_number(
                _get_or_default_and_warn(
                    sparsification,
                    CLUSTER_COUPLING_KEY,
                    DEFAULT_COUPLING_AMPLITUDE,
                    "config",
                )
            )
            max_degree = _analyse_maximal_degree(
                _get_or_default_and_warn(
                    sparsification,
                    MAXIMAL_DEGREE_KEY,
                    DEFAULT_MAXIMAL_DEGREE,
                    SPARSIFICATION_KEY,
                )
            )
            nodes = _get_or_default_and_warn(config, NODES_KEY, DEFAULT_NODES, "config")
            edges = _get_or_raise(config, EDGES_KEY, "config")
            default_field = _analyse_number(
                _get_or_default_and_warn(
                    config,
                    DEFAULT_FIELD_KEY,
                    DEFAULT_DEFAULT_FIELD,
                    "config",
                )
            )
            problem = Problem(
                edges,
                nodes,
                default_field,
            )
            node_to_childs = problem.sparsify_problem(max_degree, ampl)
            sparse_config = problem.release()
            log.info(f"Graph sparsification with {CLUSTER_COUPLING_KEY}={ampl} and {MAXIMAL_DEGREE_KEY}={max_degree} has been performed")
            return config | sparse_config, node_to_childs
        else:
            raise ConfigSyntaxError(f"`{SPARSIFICATION_KEY}` must be a dict, but got type {type(sparsification)}")
    else:
        raise ConfigSyntaxError(f"Config must be a dict, but got type {type(config)}")


def collapse_clusters(solution, node_to_childs, original_size):
    collapsed_solution = solution[:original_size]
    for par, chlds in node_to_childs.items():
        votes = {solution[par]: 1}
        for chld in chlds:
            value = solution[chld]
            if value in votes:
                votes[value] += 1
            else:
                votes[value] = 1
        if len(votes) > 1:
            log.warning(f"Conflicting values encountered when collapsing replicated nodes {[par, *chlds]}, resolved via majority vote.")
        result, _ = max(votes.items(), key = lambda x: x[1])
        collapsed_solution[par] = result
    return collapsed_solution


def postprocess(result, info):
    if info is None:
        return result
    size = info["size"]
    node_to_childs = info["node_to_childs"]
    for record in result:
        if record[0] == BLOCH_VECTORS_KEY:
            raise ValueError(f"{GET_BLOCH_VECTORS} is forbidden when graph sparsification is used")
        if record[0] == MEASUREMENT_OUTCOMES_KEY:
            record[1] = collapse_clusters(record[1], node_to_childs, size)
        else:
            assert False
    return result

