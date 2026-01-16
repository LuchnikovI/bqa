from functools import reduce, cache
from math import prod
import logging
from typing import Iterable, Iterator
import numpy as np
from numpy.random import Generator, default_rng
from numpy.typing import NDArray
from bqa.backends import Tensor
from bqa.config.config_canonicalization import Context
from bqa.config.core import config_to_context
from bqa.state import State, _initialize_state, _apply_z_layer, get_density_matrices, _run_bp, _set_to_vidal_gauge, _set_to_symmetric_gauge
from bqa.utils import NP_DTYPE

# Note that Vidal gauge breakes when maximal rank of TN is overestimated
# this is due to the pinv of sqrt(m)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_tensor(state: State, context: Context, node_id: int) -> NDArray:
    degree, pos = context.path_to_tensors[node_id]
    return state.degree_to_tensor[degree].numpy[pos]


def get_lmbd(state: State, context: Context, edge) -> NDArray:
    return state.lmbds.numpy[context.edge_to_lmbd_pos[edge]]


def get_msg(state: State, context: Context, edge) -> NDArray:
    return state.msgs.numpy[context.edge_to_msg_pos[edge]]


def get_contracted_branches(
        state: State,
        context: Context,
        is_vidal_gauge: bool = False,
):
    graph = context.graph

    def apply_msg(t: NDArray, pm: tuple[int, NDArray]) -> NDArray:
        p, m = pm
        rank = len(t.shape)
        indices_order = (*range(p), rank - 1, *range(p, rank - 1))
        mt = np.tensordot(t, m, ((p,), (1,)))
        return mt.transpose(indices_order)

    def apply_lmbd(msg: NDArray, edge) -> NDArray:
        lmbd = get_lmbd(state, context, edge)
        return msg * lmbd[np.newaxis] * lmbd[:, np.newaxis]

    @cache
    def get_msg(edge) -> NDArray:
        src_id, dst_id = edge
        tensor = get_tensor(state, context, src_id)
        input_msgs = tuple((pos + 1, apply_lmbd(get_msg((s, src_id)), (s, src_id)) if is_vidal_gauge else get_msg((s, src_id)))
                           for pos, s in enumerate(graph[src_id]) \
                           if s != dst_id)
        contraction_indices = (0, *map(lambda x: x[0], input_msgs))

        msgs_tensor = reduce(apply_msg, input_msgs, tensor)
        msg = np.tensordot(tensor.conj(), msgs_tensor, (contraction_indices, contraction_indices))
        return msg / np.linalg.trace(msg)

    return {(src_id, dst_id) : get_msg((src_id, dst_id)) \
            for src_id, ns in enumerate(graph) \
            for dst_id in ns}


def contract_tree(state: State, context: Context, is_vidal_gauge: bool = False) -> NDArray:
    graph = context.graph

    def contract_branch(edge) -> tuple[tuple[int, ...], NDArray]:
        src_id, dst_id = edge
        tensor = get_tensor(state, context, src_id)
        rank = len(tensor.shape)
        free_idx = graph[src_id].index(dst_id) + 1
        tensor = tensor.transpose((free_idx, *filter(lambda i: i != free_idx, range(rank))))
        if is_vidal_gauge:
            lmbd = get_lmbd(state, context, (src_id, dst_id)).reshape((-1,) + (rank - 1) * (1,))
            tensor = tensor * lmbd

        def gen_branches() -> Iterable[tuple[tuple[int, ...], NDArray]]:
            for s in graph[src_id]:
                if s != dst_id:
                    indices_order, branch = contract_branch((s, src_id))
                    yield indices_order, branch

        def contract_with_branch(
                acc: tuple[tuple[int, ...], NDArray],
                ids_branch: tuple[tuple[int, ...], NDArray],
        ) -> tuple[tuple[int, ...], NDArray]:
            t_indices, tensor = acc
            b_indices, branch = ids_branch
            return t_indices + b_indices, np.tensordot(tensor, branch, ((2,), (0,)))

        return reduce(contract_with_branch, gen_branches(), ((src_id,), tensor))

    dst_id = next(pos for pos, ns in enumerate(graph) if len(ns) == 1)
    src_id = graph[dst_id][0]
    edge = (src_id, dst_id)
    fwd_ids, fwd_branch = contract_branch(edge)
    ids = (dst_id,) + fwd_ids
    tensor = np.tensordot(get_tensor(state, context, dst_id), fwd_branch, axes=1)
    indices = tuple(ids.index(i) for i in range(len(ids)))
    return tensor.transpose(indices) / np.linalg.norm(tensor)


def get_density_matrices_from_tensor(tensor: NDArray) -> NDArray:

    def gen_dens() -> Iterable[NDArray]:
        rank = len(tensor.shape)
        for pos in range(rank):
            indices = tuple(i for i in range(rank) if i != pos)
            dens = np.tensordot(tensor, tensor.conj(), (indices, indices))
            dens /= np.trace(dens)
            yield dens[np.newaxis]

    return np.concatenate(list(gen_dens()), 0)


def split_nodes_by_edge(
        context: Context,
        edge,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    graph = context.graph
    src_id, dst_id = edge

    def go_to_leafs(si: int, di: int) -> Iterator[int]:
        yield di
        for n in graph[di]:
            if n != si:
                yield from go_to_leafs(di, n)

    return (tuple(go_to_leafs(src_id, dst_id)), tuple(go_to_leafs(dst_id, src_id)))


def get_tree_lmbds(context: Context, tensor: NDArray):
    lmbds = {}
    graph = context.graph
    for src_id, ns in enumerate(graph):
        for dst_id in ns:
            lmbd = lmbds.get((dst_id, src_id))
            if lmbd is None:
                lhs_inds, rhs_inds = split_nodes_by_edge(context, (src_id, dst_id))
                lhs_rank = len(lhs_inds)
                new_inds_order = lhs_inds + rhs_inds
                splitted_tensor = tensor.transpose(new_inds_order)
                shape = splitted_tensor.shape
                lhs_dim = prod(shape[:lhs_rank])
                matrix = splitted_tensor.reshape((lhs_dim, -1))
                lmbd = np.linalg.svdvals(matrix)
                lmbd /= np.linalg.norm(lmbd)
            lmbds[(src_id, dst_id)] = lmbd
    return lmbds


def randomized_tensors_in_state(state: State, rng: Generator) -> None:

    def gen_tensor(tensor: Tensor) -> Tensor:
        shape = tensor.numpy.shape
        dtype = tensor.numpy.dtype
        re = rng.normal(size = shape)
        im = rng.normal(size = shape)
        np_tensor = (re + 1j * im).astype(dtype)
        np_tensor /= np.linalg.norm(np_tensor)
        return type(tensor).make_from_numpy(np_tensor)

    new_degree_to_tensor = {degree : gen_tensor(tensor) for degree, tensor in state.degree_to_tensor.items()}
    state.degree_to_tensor = new_degree_to_tensor


SMALL_TREE_CONFIG = {
    "nodes" : {1 : 0.3, 3 : -0.7, 5 : 1., 6 : -1., 7 : 0.25},
    "edges" : {(2, 0) : 1.,
               (1, 2) : -1.,
               (2, 4) : 0.5,
               (4, 3) : -0.5,
               (4, 5) : 0.75,
               (4, 6) : -0.75,
               (6, 7) : 0.3,
               (8, 6) : 0.6},
    "bp_eps" : 1e-10,
    "pinv_eps" : 1e-7,
    "default_field" : 0.6,
}

def test_bp_small_tree_circuit():
    rng = default_rng(42)
    context = config_to_context(SMALL_TREE_CONFIG)
    state = _initialize_state(context)
    _apply_z_layer(context, 0., state)
    _apply_z_layer(context, 0., state)
    _apply_z_layer(context, 0., state)
    _apply_z_layer(context, 0., state)
    randomized_tensors_in_state(state, rng)
    _run_bp(context, state)
    msgs_correct = get_contracted_branches(state, context)
    for edge, msg_correct in msgs_correct.items():
        assert np.isclose(msg_correct, get_msg(state, context, edge)).all()
    print("test_bp_small_tree_circuit: OK")


def test_density_matrices_small_tree_circuit():
    rng = default_rng(42)
    context = config_to_context(SMALL_TREE_CONFIG)
    state = _initialize_state(context)
    _apply_z_layer(context, 0., state)
    _apply_z_layer(context, 0., state)
    _apply_z_layer(context, 0., state)
    _apply_z_layer(context, 0., state)
    randomized_tensors_in_state(state, rng)
    _run_bp(context, state)
    dens = get_density_matrices(context, state)
    correct_dens = get_density_matrices_from_tensor(contract_tree(state, context))
    assert np.isclose(dens, correct_dens).all()
    print("test_density_matrices_small_tree_circuit: OK")


def test_vg_and_sg_small_tree_circuit():
    rng = default_rng(43)
    context = config_to_context(SMALL_TREE_CONFIG)
    state = _initialize_state(context)
    _apply_z_layer(context, 0., state)
    _apply_z_layer(context, 0., state)
    _apply_z_layer(context, 0., state)
    _apply_z_layer(context, 0., state)
    randomized_tensors_in_state(state, rng)
    tensor = contract_tree(state, context)
    correct_lmbds = get_tree_lmbds(context, tensor)
    _run_bp(context, state)
    dens_before_canonicalization = get_density_matrices(context, state)
    _set_to_vidal_gauge(context, state)
    branches = get_contracted_branches(state, context, is_vidal_gauge=True)
    for edge, correct_lmbd in correct_lmbds.items():
        lmbd = get_lmbd(state, context, edge)
        branch = branches[edge]
        r = min(correct_lmbd.shape[0], lmbd.shape[0])
        assert np.isclose(branch[r:], 0.).all()
        assert np.isclose(branch[:, r:], 0.).all()
        assert np.isclose(branch[:r, :r], np.eye(r) / r).all()
        assert np.isclose(correct_lmbd[:r], lmbd[:r]).all()
        assert np.isclose(np.zeros((1,), NP_DTYPE), lmbd[r:]).all()
        assert np.isclose(np.zeros((1,), NP_DTYPE), correct_lmbd[r:]).all()
    tensor_vg = contract_tree(state, context, True)
    assert np.isclose(tensor, tensor_vg).all()
    _set_to_symmetric_gauge(context, state)
    _run_bp(context, state)
    tensor_sg = contract_tree(state, context)
    assert np.isclose(tensor, tensor_sg).all()
    dens_after_canonicalization = get_density_matrices(context, state)
    assert np.isclose(dens_before_canonicalization, dens_after_canonicalization).all()
    print("test_vg_small_tree_circuit: OK")


