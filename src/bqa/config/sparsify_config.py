import logging
from copy import copy
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from bqa.config.desugar_config import canonicalize_edge_id
from bqa.config.validate_config import (CLUSTER_COUPLING_AMPLITUDE_KEY, DEFAULT_FIELD_KEY, EDGES_KEY, NODES_KEY, POSTPROCESSING_KEY,
                                        SPARSIFICATION_KEY, EPS_KEY)
from bqa.config.pipeline import pipeline

log = logging.getLogger(__name__)

BLOCH_VECTORS_KEY = "bloch_vectors"

MEASUREMENT_OUTCOMES_KEY = "measurement_outcomes"

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
            return other.coupling < self.coupling
        elif self.is_leaf != other.is_leaf:
            return self.is_leaf
        else:
            return id(self) < id(other)


def node_to_tree(couplings_and_neighs, ampl):
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


class DynGraph:
    def __init__(self, edges, nodes, default_field):
        self.default_field = default_field
        self.nodes = copy(nodes)
        self.edges = copy(edges)
        self.graph_size = max(
            max(self.nodes.keys(), default = -1),
            max((max(i, j) for i, j in self.edges.keys()), default = -1)
        ) + 1
        self.graph = [set() for _ in range(self.graph_size)]
        for i, j in self.edges.keys():
            self.graph[i].add(j)
            self.graph[j].add(i)

    def get_field(self, node_id):
        assert node_id < self.graph_size
        field = self.nodes.get(node_id, self.default_field)
        return field

    def get_degree(self, node_id):
        assert node_id < self.graph_size
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
        remaining_weight = eps * total_weight
        queue = []
        seen = set()
        number_of_removed_edges = 0
        for edge_id, coupling in self.edges.items():
            if edge_id not in seen:
                heappush(queue, (abs(coupling), -self.get_degree(edge_id[0]), edge_id))
                seen.add(edge_id)
        while queue:
            coupling, neg_degree, edge_id = heappop(queue)
            real_degree = min(self.get_degree(edge_id[0]), self.get_degree(edge_id[1]))
            if real_degree != -neg_degree:
                heappush(queue, (coupling, -real_degree, edge_id))
                continue
            if remaining_weight - coupling < 0:
                break
            remaining_weight -= coupling
            self.remove_edge(*edge_id)
            number_of_removed_edges += 1
        log.info(f"{number_of_removed_edges} edges has been removed, total relative accuracy drop {(eps * total_weight - remaining_weight) / total_weight} < {eps}")

    def compile_to_degree_three(self, ampl):
        postprocessing_info = {"graph_size": self.graph_size, "node_to_aux_nodes" : {}}
        node_to_aux_nodes = postprocessing_info["node_to_aux_nodes"]
        old_nodes_number = self.graph_size
        for node_id in range(old_nodes_number):
            if self.get_degree(node_id) > 3:
                id_to_coupling = {i : c for c, i in self.get_couplings_and_ids(node_id)}
                tree = node_to_tree(self.get_couplings_and_ids(node_id), ampl)
                self.disconnect_node(node_id)
                stack = [(node_id, tree)]
                node_to_aux_nodes[node_id] = []
                while stack:
                    par_node_id, node = stack[-1]
                    stack.pop()
                    assert not isinstance(node.content, int)
                    for sub_tree in node.content:
                        if sub_tree.is_leaf:
                            aux_node_id = sub_tree.content
                            assert isinstance(aux_node_id, int)
                            coupling = id_to_coupling[aux_node_id]
                            self.add_edge(aux_node_id, par_node_id, coupling)
                        else:
                            ch_node_id = self.add_node(0.)
                            node_to_aux_nodes[node_id].append(ch_node_id)
                            coupling = sub_tree.coupling
                            assert coupling <= 0
                            stack.append((ch_node_id, sub_tree))
                            self.add_edge(ch_node_id, par_node_id, coupling)
        new_nodes_number = self.graph_size
        log.info(f"Graph has been compiled to minimal possible degree, initial nodes number {old_nodes_number}, new nodes number {new_nodes_number}")
        return postprocessing_info

    @property
    def nodes_and_edges(self):
        return self.nodes, self.edges


@pipeline
def sparsify_config(config):
    sparsification = config[SPARSIFICATION_KEY]
    if sparsification is None:
        return config
    nodes = config[NODES_KEY]
    edges = config[EDGES_KEY]
    default_field = config[DEFAULT_FIELD_KEY]
    eps = sparsification[EPS_KEY]
    ampl = sparsification[CLUSTER_COUPLING_AMPLITUDE_KEY]
    dyn_graph = DynGraph(edges, nodes, default_field)
    dyn_graph.sparsify(eps)
    postprocessing_info = dyn_graph.compile_to_degree_three(ampl)
    nodes, edges = dyn_graph.nodes_and_edges
    new_config = {
        NODES_KEY : nodes,
        EDGES_KEY : edges,
        POSTPROCESSING_KEY : postprocessing_info,
        **{k : v for k, v in config.items() if k not in {NODES_KEY, EDGES_KEY, SPARSIFICATION_KEY}},
    }
    return new_config


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
    size = info["graph_size"]
    node_to_aux_nodes = info["node_to_aux_nodes"]
    for record in result:
        if record[0] == MEASUREMENT_OUTCOMES_KEY:
            record[1] = collapse_clusters(record[1], node_to_aux_nodes, size)
        else:
            assert False
    return result
