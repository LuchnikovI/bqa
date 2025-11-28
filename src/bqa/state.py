from dataclasses import dataclass
import logging
import numpy as np
from functools import reduce
from typing import Iterable
from numpy.random import Generator
from numpy.typing import NDArray
from bqa.backends import Tensor
from bqa.config.config_canonicalization import Context
from bqa.config.config_canonicalization import Layout
from bqa.utils import dict_reduce, dispatch_precision, dict_map

log = logging.getLogger(__name__)

# state initialization

@dataclass
class State:
    degree_to_tensor: dict[int, Tensor]
    msgs: Tensor
    lmbds: Tensor

def _make_batch(arr: NDArray, batch_size: int) -> NDArray:
    shape = arr.shape
    return np.broadcast_to(arr, (batch_size, *shape))

def _make_initial_degree_to_tensor(context: Context) -> dict[int, Tensor]:
    dtype = dispatch_precision(np.complex64, np.complex128)
    backend = context.backend
    local_state = np.array([np.sqrt(0.5), -np.sqrt(0.5)], dtype)

    def get_initial_tensors_batch(degree: int, layout: Layout) -> Tensor:
        shape = (2,) + (1,) * degree
        batch_size = layout.node_ids.batch_size
        single_tensor = local_state.reshape(shape)
        return backend.from_numpy(_make_batch(single_tensor, batch_size))

    return dict_map(get_initial_tensors_batch, context.degree_to_layout)

def _make_msgs_from_lmbds(lmbds: Tensor, context: Context) -> Tensor:
    return lmbds.batch_slice(context.msg_pos_to_lmbd_pos).batch_normalize().batch_diag()

def _make_initial_lmbds(context: Context) -> Tensor:
    dtype = dispatch_precision(np.complex64, np.complex128)
    backend = context.backend
    lmbds_number = context.lmbds_number
    ones_bacthed = _make_batch(np.ones((1,), dtype), lmbds_number)
    return backend.from_numpy(ones_bacthed)

def _initialize_state(context: Context) -> State:
    lmbds = _make_initial_lmbds(context)
    return State(_make_initial_degree_to_tensor(context),
                 _make_msgs_from_lmbds(lmbds, context),
                 lmbds)

# density matrices

def _get_density_matrices(context: Context, state: State) -> NDArray:

    def get_tensors_aligned_input_msgs(layout: Layout) -> tuple[Tensor, ...]:
        return tuple(map(lambda idx: state.msgs.batch_slice(idx), layout.input_msgs_position))

    def add_density_matrices_for_degree(dms: NDArray, degree: int, layout: Layout) -> NDArray:
        input_msgs = get_tensors_aligned_input_msgs(layout)
        tensors = state.degree_to_tensor[degree]
        src = tensors.get_density_matrices(input_msgs).numpy
        node_ids_np = layout.node_ids.numpy
        dms[node_ids_np] = src
        return dms

    dtype = dispatch_precision(np.complex64, np.complex128)
    density_matrices = np.empty((context.nodes_number, 2, 2), dtype)
    return dict_reduce(add_density_matrices_for_degree, context.degree_to_layout, density_matrices)

def _run_bp(context: Context, state: State) -> State:
    raise NotImplementedError()

def _apply_z_layer(context: Context, ztime: float, state: State) -> State:
    raise NotImplementedError()

def _apply_x_layer(xtime: float, state: State) -> State:
    raise NotImplementedError()

def _truncate_vidal_gauge(context: Context, state: State) -> State:
    raise NotImplementedError()

def _set_to_vidal_gauge(context: Context, state: State) -> State:
    raise NotImplementedError()

def _set_to_symmetric_gauge(context: Context, state: State) -> State:
    state.msgs = _make_msgs_from_lmbds(state.lmbds, context)

    def get_tensors_aligned_sqrt_lmbds(layout: Layout) -> tuple[Tensor, ...]:
        return tuple(map(lambda idx: state.lmbds.batch_slice(idx).sqrt(), layout.lmbds_position))

    def modify_tensors_per_degree(state: State, degree: int, layout: Layout) -> State:
        sqrt_lmbds = get_tensors_aligned_sqrt_lmbds(layout)
        updated_tensors = state.degree_to_tensor[degree].mul_by_lmbds(sqrt_lmbds)
        state.degree_to_tensor[degree] = updated_tensors
        return state

    return dict_reduce(modify_tensors_per_degree, context.degree_to_layout, state)

def measure(context: Context, state: State) -> list:
    measurement_outcomes = {}
    threshold = context.measurement_threshold

    def get_non_measured_ground_probs(state: State) -> list[tuple[int, float]]:
        return list(map(lambda pair: (pair[0], float(pair[1][0, 0])),
                        filter(lambda pair: pair[0] not in measurement_outcomes,
                               enumerate(_get_density_matrices(context, state)))))

    def get_z_abs_value(node_to_gp: tuple[int, float]) -> float:
        return abs(2 * node_to_gp[1] - 1)

    def sample_outcome(p0: float, rng: Generator) -> int:
        val = rng.uniform(0., 1.)
        return 0 if p0 > val else 1

    def apply_and_register_measurement(
            state: State,
            projector_id: int,
            node_id: int,
    ) -> State:
        measurement_outcomes[node_id] = projector_id
        raise NotImplementedError()

    def apply_and_register_0_measurement(state: State, node_id: int) -> State:
        return apply_and_register_measurement(state, 0, node_id)

    def apply_and_register_1_measurement(state: State, node_id: int) -> State:
        return apply_and_register_measurement(state, 1, node_id)

    def get_nodes_above_threshold(node_to_gp: list[tuple[int, float]]) -> Iterable[int]:
        return map(lambda pair: pair[0],
                   filter(lambda pair: pair[1] > threshold,
                          node_to_gp))

    def get_nodes_below_threshold(node_to_gp: list[tuple[int, float]]) -> Iterable[int]:
        return map(lambda pair: pair[0],
                   filter(lambda pair: pair[1] < 1. - threshold,
                          node_to_gp))

    while len(measurement_outcomes) < context.nodes_number:
        node_to_gp = get_non_measured_ground_probs(state)
        node_id, p0 = min(node_to_gp, key=get_z_abs_value)
        projector_id = sample_outcome(p0, context.numpy_rng)
        state = apply_and_register_measurement(state, projector_id, node_id)
        state = _run_bp(context, state)
        state = reduce(apply_and_register_0_measurement, get_nodes_above_threshold(node_to_gp), state)
        state = _run_bp(context, state)
        state = reduce(apply_and_register_1_measurement, get_nodes_below_threshold(node_to_gp), state)
        state = _run_bp(context, state)
    state = _set_to_vidal_gauge(context, state)
    state = _truncate_vidal_gauge(context, state)
    state = _set_to_symmetric_gauge(context, state)
    _ = _run_bp(context, state)
    measurement_outcomes = list(map(lambda node_id: measurement_outcomes[node_id], range(len(measurement_outcomes))))
    log.info("Sampling measurement outcomes completed")
    return measurement_outcomes

def run_layer(context: Context, xtime: float, ztime: float, state: State) -> None:
    state = _apply_z_layer(context, ztime, state)
    state = _run_bp(context, state)
    state = _set_to_vidal_gauge(context, state)
    state = _truncate_vidal_gauge(context, state)
    state = _set_to_symmetric_gauge(context, state)
    state = _run_bp(context, state)
    _ = _apply_x_layer(xtime, state)  # in all subroutines state is also mutated accordingly
    log.info(f"Layer with ztime {ztime} and xtime {xtime} is applied")

