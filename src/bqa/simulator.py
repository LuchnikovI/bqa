from dataclasses import dataclass
from functools import reduce
import numpy as np
from numpy.typing import NDArray
from bqa.context import Context, Layout
from bqa.backends import Tensor

LOCAL_DIM = 2
NP_DTYPE = np.complex128

# degree_to_tensor abstraction

def _make_degree_to_tensor(local_states: NDArray, context: Context) -> dict[int, Tensor]:
    if local_states.shape[1] != LOCAL_DIM:
        raise ValueError(f"Local dimensions must be {LOCAL_DIM}, got {local_states.shape[1]}")
    local_states_tensor = context.backend.from_numpy(local_states)
    def turn_to_aligned_tensor(degree_to_layout: tuple[int, Layout]):
        degree, layout = degree_to_layout
        tensor = local_states_tensor.batch_slice(layout.node_ids)
        return degree, tensor.batch_reshape((-1, *(degree * (1,)))).batch_normalize()
    return dict(map(turn_to_aligned_tensor, context.degree_to_layout.items()))

# lmbds abstraction

def _make_batch_by_copying(arr: NDArray, batch_size: int) -> NDArray:
    return arr[np.newaxis].repeat(batch_size, 0)

def _make_initial_lmbds(context: Context) -> Tensor:
    backend = context.backend
    lmbds_number = context.lmbds_number
    ones_bacthed = _make_batch_by_copying(np.ones((1,), dtype=NP_DTYPE), lmbds_number)
    return backend.from_numpy(ones_bacthed)

# msgs abstraction

def _make_msgs(lmbds: Tensor, context: Context) -> Tensor:
    return lmbds.batch_slice(context.msg_pos_to_lmbd_pos).batch_normalize().batch_diag()

# state abstraction

@dataclass
class State:
    degree_to_tensor: dict[int, Tensor]
    msgs: Tensor
    lmbds: Tensor


def make_product_initial_state(local_states: NDArray, context: Context) -> State:
    if local_states.shape[0] != context.qubits_number:
        raise ValueError(f"Number of qubits in the system is {context.qubits_number} but given local states for {local_states.shape[0]} number of qubits")
    lmbds = _make_initial_lmbds(context)
    return State(_make_degree_to_tensor(local_states, context),
                 _make_msgs(lmbds, context),
                 lmbds)

def make_uniform_product_initial_state(local_state: NDArray, context: Context) -> State:
    return make_product_initial_state(_make_batch_by_copying(local_state, context.qubits_number), context)

def make_standard_annealing_initial_state(context: Context) -> State:
    raise NotImplementedError()

def compute_density_matrices(state: State, context: Context) -> NDArray:
    qubits_number = context.qubits_number
    degree_to_tensor = state.degree_to_tensor
    degree_to_layout = context.degree_to_layout
    msgs = state.msgs
    def get_density_matrix(degree_and_layout: tuple[int, Layout]) -> tuple[int, NDArray]:
        degree, layout = degree_and_layout
        tensor = degree_to_tensor[degree]
        aligned_msgs = tuple(map(lambda poss: msgs.batch_slice(poss), layout.input_msgs_pos))
        return degree, tensor.get_density_matrices(aligned_msgs).numpy
    def collect_density_matrices_into_array(dst: NDArray, degree_and_density_matrices: tuple[int, NDArray]):
        degree, density_matrices = degree_and_density_matrices
        poss = degree_to_layout[degree].node_ids.numpy
        dst[poss] = density_matrices
        return dst
    density_matrices_per_degree = map(get_density_matrix, degree_to_layout.items())
    return reduce(
        collect_density_matrices_into_array,
        density_matrices_per_degree,
        np.empty((qubits_number, LOCAL_DIM, LOCAL_DIM), dtype = np.complex128))

