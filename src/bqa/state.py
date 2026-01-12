from dataclasses import dataclass
import logging
import numpy as np
from typing import Iterator
from numpy.random import Generator, default_rng
from numpy.typing import NDArray
from bqa.backends import Tensor
from bqa.config.config_canonicalization import Context
from bqa.config.config_canonicalization import Layout
from bqa.utils import NP_DTYPE

# for efficiency reasons this code is highly imperative,
# one can think of it as a virtual machine for quantum
# annealing execution

log = logging.getLogger(__name__)

# state initialization

INITIAL_STATE = np.array([np.sqrt(0.5), -np.sqrt(0.5)], NP_DTYPE)

# this is being mutated in many places
@dataclass
class State:
    rng: Generator
    degree_to_tensor: dict[int, Tensor]
    msgs: Tensor
    lmbds: Tensor

    @property
    def bond_dim(self) -> int:
        return next(iter(self.degree_to_tensor.items()))[1].batch_shape[-1]


def _make_batch(arr: NDArray, batch_size: int) -> NDArray:
    shape = arr.shape
    return np.broadcast_to(arr, (batch_size, *shape))


def _make_initial_degree_to_tensor(context: Context) -> dict[int, Tensor]:
    backend = context.backend

    def get_initial_tensors_batch(degree: int, layout: Layout) -> Tensor:
        shape = (2,) + (1,) * degree
        batch_size = layout.node_ids.batch_size
        single_tensor = INITIAL_STATE.reshape(shape)
        return backend.make_from_numpy(_make_batch(single_tensor, batch_size))

    return {
        degree: get_initial_tensors_batch(degree, layout)
        for degree, layout in context.degree_to_layout.items()
    }


def _make_msgs_from_lmbds(lmbds: Tensor, context: Context) -> Tensor:
    return lmbds.batch_slice(context.msg_pos_to_lmbd_pos).batch_diag().batch_trace_normalize()


def _make_initial_lmbds(context: Context) -> Tensor:
    backend = context.backend
    lmbds_number = context.lmbds_number
    ones_bacthed = _make_batch(np.ones((1,), NP_DTYPE), lmbds_number)
    return backend.make_from_numpy(ones_bacthed)


def _initialize_state(context: Context) -> State:
    lmbds = _make_initial_lmbds(context)
    return State(
        default_rng(context.seed),
        _make_initial_degree_to_tensor(context),
        _make_msgs_from_lmbds(lmbds, context),
        lmbds,
    )


# density matrices


def get_density_matrices(context: Context, state: State) -> NDArray:
    density_matrices = np.empty((context.nodes_number, 2, 2), NP_DTYPE)

    def set_densities(node_ids: NDArray, src_density: NDArray) -> None:
        density_matrices[node_ids] = src_density

    def get_node_ids_and_density_matrices_iter() -> Iterator[tuple[NDArray, NDArray]]:
        for degree, layout in context.degree_to_layout.items():
            tensors = state.degree_to_tensor[degree]
            input_msgs = tuple(
                state.msgs.batch_slice(idx) for idx in layout.input_msgs_position
            )
            yield layout.node_ids.numpy, tensors.get_density_matrices(input_msgs).numpy

    for node_ids, src_density in get_node_ids_and_density_matrices_iter():
        set_densities(node_ids, src_density)
    log.info("Density matrices have been computed")
    return density_matrices


def _run_bp(context: Context, state: State) -> None:
    log.debug("BP algorithm has started")
    bond_dim = state.bond_dim
    bp_eps = context.bp_eps
    max_bp_iters = context.max_bp_iters_number
    new_msgs = context.backend.make_empty(context.edges_number, (bond_dim, bond_dim))
    for iter_num in range(context.max_bp_iters_number):
        for degree, tensor in state.degree_to_tensor.items():
            input_msgs_position = context.degree_to_layout[degree].input_msgs_position
            output_msgs_position = context.degree_to_layout[degree].output_msgs_position
            aligned_msgs = tuple(state.msgs.batch_slice(pos) for pos in input_msgs_position)
            new_output_msgs = tensor.pass_msgs(aligned_msgs)
            for ms, poss in zip(new_output_msgs, output_msgs_position):
                new_msgs.assign_at_batch_indices(ms, poss)
        dist = new_msgs.get_dist(state.msgs).numpy
        log.debug(f"Iteration {iter_num}, distance between subsequent messages {dist}")
        if dist < bp_eps:
            log.debug(f"BP algorithm completed after {iter_num} iterations")
            return
        state.msgs.make_inplace_damping_update(new_msgs, context.damping)
    log.warning(f"BP algorithm exceeds iterations limit set to {max_bp_iters}")


def _apply_z_layer(context: Context, ztime: float, state: State) -> None:

    def apply_z_to_tensor(degree: int, tensor: Tensor) -> Tensor:
        layout = context.degree_to_layout[degree]
        node_ampls = layout.node_ampls
        edge_ampls = (a * ztime for a in layout.edge_ampls)
        return tensor.apply_z_gates(ztime * node_ampls).apply_conditional_z_gates(edge_ampls)

    new_degree_to_tensor = {d : apply_z_to_tensor(d, t) for d, t in state.degree_to_tensor.items()}
    new_msgs = state.msgs.extend_msgs()
    state.degree_to_tensor = new_degree_to_tensor
    state.msgs = new_msgs


def _apply_x_layer(xtime: float, state: State) -> None:
    new_degree_to_tensor = {d : t.apply_x_gates(xtime) for d, t in state.degree_to_tensor.items()}
    state.degree_to_tensor = new_degree_to_tensor


def _truncate_vidal_gauge(context: Context, state: State) -> None:
    bond_dim = context.max_bond_dim
    truncated_degree_to_tensor = {d : t.batch_truncate_all_but(bond_dim, [0]) for d, t in state.degree_to_tensor.items()}
    truncated_lmbds = state.lmbds.batch_truncate_all_but(bond_dim)
    state.degree_to_tensor = truncated_degree_to_tensor
    state.lmbds = truncated_lmbds
    new_tensor_shapes = {d : t.batch_shape for d, t in state.degree_to_tensor.items()}
    new_lmbds_shapes = state.lmbds.batch_shape
    log.debug(f"Vidal gauge has been truncated, tensor shapes: {new_tensor_shapes}, lmbds shapes: {new_lmbds_shapes}")


def _set_to_vidal_gauge(context: Context, state: State) -> None:
    pinv_eps = context.pinv_eps

    # these three functions rely on the insertion order of msgs into a dict
    def get_fwd(t: Tensor) -> Tensor:
        return t.get_batch_slice(range(context.lmbds_number))

    def get_bwd(t: Tensor) -> Tensor:
        return t.get_batch_slice(range(context.lmbds_number, context.edges_number))

    def assembly_canonicalizers(fwd_canonicalizer: Tensor, bwd_canonicalizer: Tensor) -> Tensor:
        return bwd_canonicalizer.batch_concat(fwd_canonicalizer)
    # ---------------------------------------------------------------

    def colide(fwd: Tensor, bwd: Tensor) -> tuple[Tensor, ...]:
        ker = fwd.batch_tensordot(bwd, [[1], [1]])
        us, s, vhs = ker.get_batch_svd(pinv_eps)
        return us, s, vhs.batch_transpose((1, 0))

    ul, lu = state.msgs.decompose_iden_using_msgs(pinv_eps)
    fwd_lu = get_fwd(lu)
    bwd_lu = get_bwd(lu)
    fwd_ul = get_fwd(ul)
    bwd_ul = get_bwd(ul)
    us, lmbds, vs = colide(fwd_lu, bwd_lu)
    state.lmbds = lmbds.batch_normalize()
    fwd_canonicalizer = fwd_ul.batch_matmul(us)
    bwd_canonicalizer = bwd_ul.batch_matmul(vs)
    canonicalizers = assembly_canonicalizers(fwd_canonicalizer, bwd_canonicalizer)
    degree_to_tensor = state.degree_to_tensor
    for degree, layout in context.degree_to_layout.items():
        aligned_canonicalizers = tuple(
            canonicalizers.batch_slice(idx) for idx in layout.input_msgs_position
        )
        degree_to_tensor[degree] = degree_to_tensor[degree].apply_canonicalizers(
            aligned_canonicalizers
        )
    log.debug("State has been set to Vidal gauge")


def _set_to_symmetric_gauge(context: Context, state: State) -> None:
    state.msgs = _make_msgs_from_lmbds(state.lmbds, context)
    for degree, layout in context.degree_to_layout.items():
        sqrt_lmbds = tuple(
            state.lmbds.batch_slice(idx).sqrt() for idx in layout.lmbds_position
        )
        updated_tensors = state.degree_to_tensor[degree].mul_by_lmbds(sqrt_lmbds)
        state.degree_to_tensor[degree] = updated_tensors
    log.debug("State has been set to Symmetric_gauge")


def measure(context: Context, state: State) -> list:
    measurement_outcomes = {}
    threshold = context.measurement_threshold
    rng = state.rng

    def is_not_all_measured() -> bool:
        return len(measurement_outcomes) < context.nodes_number

    def get_not_measured_ground_probs() -> dict[int, float]:
        return {
            node_id: dens[0, 0]
            for node_id, dens in enumerate(get_density_matrices(context, state))
            if node_id not in measurement_outcomes
        }

    def get_z_abs_value(node_id_prob: tuple[int, float]) -> float:
        return abs(2 * node_id_prob[1] - 1)

    def sample_outcome(prob: float) -> int:
        val = rng.uniform(0.0, 1.0)
        return 0 if prob > val else 1

    def apply_and_register_measurement(
        measurement_outcome: int,
        node_id: int,
    ) -> None:
        measurement_outcomes[node_id] = 1 - 2 * measurement_outcome
        degree, pos = context.path_to_tensors[node_id]
        state.degree_to_tensor[degree].measure(pos, measurement_outcome)

    def apply_and_register_0_measurement(node_id: int) -> None:
        apply_and_register_measurement(0, node_id)

    def apply_and_register_1_measurement(node_id: int) -> None:
        apply_and_register_measurement(1, node_id)

    def measure_nodes_above_threshold(not_measured_nodes: dict[int, float]) -> None:
        above_thrshld = list(node_id for node_id, prob in not_measured_nodes.items() if prob > threshold)
        for node_id in above_thrshld:
            apply_and_register_0_measurement(node_id)
        log.debug(f"Nodes {above_thrshld} were above threshold and have been projected up")

    def measure_nodes_below_threshold(not_measured_nodes: dict[int, float]) -> None:
        below_thrshld = list(node_id for node_id, prob in not_measured_nodes.items() if prob < 1.0 - threshold)
        for node_id in below_thrshld:
            apply_and_register_1_measurement(node_id)
        log.debug(f"Nodes {below_thrshld} were below threshold and have been projected down")

    while is_not_all_measured():
        not_measured_ground_probs = get_not_measured_ground_probs()
        node_id, prob = min(not_measured_ground_probs.items(), key=get_z_abs_value)
        measurement_outcome = sample_outcome(prob)
        log.debug(f"Node {node_id} had highest uncertainty and has been measured")
        apply_and_register_measurement(measurement_outcome, node_id)
        _run_bp(context, state)
        not_measured_ground_probs = get_not_measured_ground_probs()
        measure_nodes_above_threshold(not_measured_ground_probs)
        measure_nodes_below_threshold(not_measured_ground_probs)
        _run_bp(context, state)
    _set_to_vidal_gauge(context, state)
    _truncate_vidal_gauge(context, state)
    _set_to_symmetric_gauge(context, state)
    _run_bp(context, state)
    measurement_outcomes = [measurement_outcomes[node_id] for node_id in range(len(measurement_outcomes))]
    log.info("Sampling measurement outcomes completed")
    return measurement_outcomes


def run_layer(context: Context, xtime: float, ztime: float, state: State) -> None:
    _apply_z_layer(context, ztime, state)
    _run_bp(context, state)
    _set_to_vidal_gauge(context, state)
    _truncate_vidal_gauge(context, state)
    _set_to_symmetric_gauge(context, state)
    _run_bp(context, state)
    _apply_x_layer(xtime, state)  # in all subroutines state is also mutated accordingly
    log.info(f"Layer with ztime {ztime} and xtime {xtime} is applied")
