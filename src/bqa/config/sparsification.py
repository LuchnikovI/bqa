import logging
from copy import copy
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from bqa.config.schedule_syntax import GET_BLOCH_VECTORS
from .utils import (
    ConfigSyntaxError, _get_or_default_and_warn, _get_or_raise,
    _analyse_number, _analyse_non_neg_int, _analyse_number_greater_1,
    _analyse_0_to_1_number)
from .config_syntax import (DEFAULT_NODES, EDGES_KEY, NODES_KEY, DEFAULT_FIELD_KEY,
                            DEFAULT_DEFAULT_FIELD, _analyse_edge_id)

log = logging.getLogger(__name__)

SPARSIFICATION_KEY = "sparsification"

CLUSTER_COUPLING_AMPLITUDE_KEY = "cluster_coupling_amplitude"

EPS_KEY = "eps"

BLOCH_VECTORS_KEY = "bloch_vectors"

MEASUREMENT_OUTCOMES_KEY = "measurement_outcomes"

DEFAULT_CLUSTER_COUPLING_AMPLITUDE = 1.1

DEFAULT_MAXIMAL_DEGREE = 3

DEFAULT_EPS = 0.0

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


@dataclass(frozen=True)
class Node:
    is_leaf: bool
    coupling: float
    content: list["Node"] | int

    def __lt__(self, other):
        assert isinstance(other, Node)
        assert self.coupling is not None
        assert other.coupling is not None
        if self.coupling != other.coupling:
            return other.coupling < self.coupling  # couplings are negative, larger couplings come first
        elif self.is_leaf != other.is_leaf:
            return self.is_leaf
        else:
            return id(self) < id(other)


def _node_to_tree(couplings_and_neighs, ampl):
    queue = [Node(True, -ampl * abs(c), i) for c, i in couplings_and_neighs]
    assert len(queue) > 3
    heapify(queue)
    while True:
        lhs = heappop(queue)
        rhs = heappop(queue)
        if not queue:
            max_tree, min_tree = (lhs, rhs) if lhs > rhs else (rhs, lhs)
            if max_tree.is_leaf:
                return Node(False, 0., [max_tree, min_tree])
            else:
                assert not isinstance(max_tree.content, int)
                return Node(False, 0., [min_tree, *(max_tree.content)])
        heappush(queue, Node(False, lhs.coupling + rhs.coupling, [lhs, rhs]))


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
        if node_id >= self.graph_size:
            assert False
        field = self.nodes.get(node_id, self.default_field)
        assert field is not None
        return field

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

    def disconnect_node(self, node_id):
        for neigh_id in copy(self.graph[node_id]):
            self.remove_edge(node_id, neigh_id)

    def move_edge(self, i, old_j, new_j):
        assert i != old_j
        assert i != new_j
        assert i < self.graph_size
        assert old_j < self.graph_size
        assert new_j < self.graph_size
        coupling = self.remove_edge(i, old_j)
        self.add_edge(i, new_j, coupling)

    def get_couplings_and_ids(self, node_id):
        for neigh_id in self.graph[node_id]:
            edge_id = canonicalize_edge_id((node_id, neigh_id))
            coupling = self.edges[edge_id]
            yield coupling, neigh_id

    def sparsify(self, eps):
        total_weight = sum(abs(coupling) for _, coupling in self.edges.items())
        drop_weight = eps * total_weight
        drop_weight_clone = drop_weight
        queue = []
        seen = set()
        number_of_removed_edges = 0
        for edge_id, coupling in self.edges.items():
            edge_id = canonicalize_edge_id(edge_id)
            if edge_id not in seen:
                heappush(queue, (abs(coupling), -self.get_degree(edge_id[0]), edge_id))
                seen.add(edge_id)
        while queue:
            coupling, _, edge_id = heappop(queue)
            if drop_weight - coupling < 0:
                break
            drop_weight -= coupling
            self.remove_edge(*edge_id)
            number_of_removed_edges += 1
        log.info(f"{number_of_removed_edges} edges has been removed, total relative accuracy drop {(drop_weight_clone - drop_weight) / total_weight} < {eps}")

    def compile_to_degree_three(self, ampl):
        info = {"size": self.graph_size, "node_to_childs" : {}}
        node_to_childs = info["node_to_childs"]
        old_nodes_number = self.graph_size
        for node_id in range(self.graph_size):
            if self.get_degree(node_id) > 3:
                couplings_and_ids = {i : c for c, i in self.get_couplings_and_ids(node_id)}
                tree = _node_to_tree(self.get_couplings_and_ids(node_id), ampl)
                self.disconnect_node(node_id)
                stack = [(node_id, tree)]
                node_to_childs[node_id] = []
                while stack:
                    par_node_id, node = stack[-1]
                    stack.pop()
                    assert not isinstance(node.content, int)
                    for sub_tree in node.content:
                        if sub_tree.is_leaf:
                            ch_node_id = sub_tree.content
                            assert isinstance(ch_node_id, int)
                            coupling = couplings_and_ids[ch_node_id]
                            self.add_edge(ch_node_id, par_node_id, coupling)
                        else:
                            ch_node_id = self.add_node(0.)
                            node_to_childs[node_id].append(ch_node_id)
                            coupling = sub_tree.coupling
                            assert coupling <= 0
                            stack.append((ch_node_id, sub_tree))
                            self.add_edge(ch_node_id, par_node_id, coupling)
        new_nodes_number = self.graph_size
        log.info(f"Graph has been compiled to minimal possible degree, initial nodes number {old_nodes_number}, new nodes number {new_nodes_number}")
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
            ampl = _analyse_number_greater_1(
                _get_or_default_and_warn(
                    sparsification,
                    CLUSTER_COUPLING_AMPLITUDE_KEY,
                    DEFAULT_CLUSTER_COUPLING_AMPLITUDE,
                    "config",
                )
            )
            eps = _analyse_0_to_1_number(
                _get_or_default_and_warn(
                    sparsification,
                    EPS_KEY,
                    DEFAULT_EPS,
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
            problem.sparsify(eps)
            node_to_childs = problem.compile_to_degree_three(ampl)
            sparse_config = problem.release()
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
            log.warning(f"Conflicting values encountered when collapsing replicated nodes {[par, *chlds]}, resolved via majority vote from {votes}.")
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

