import numpy as np
from numpy.typing import NDArray
from numpy.random import uniform
from bqa.config.config_canonicalization import Context
from bqa.config.core import config_to_context

try:
    from qem import QuantumState
except ImportError as e:
    print("Exact quantum computing simulator is not install. To instal it one needs: \n1) install rust (see https://rust-lang.org/tools/install/) \n2) install maturin (pip install maturin) \n3) install bqa with test deps (poetry install --with test).")
    raise e

FLIPPED_HADAMARD = (1. / np.sqrt(2.)) * np.array([
    [1, 1],
    [-1, 1],
], dtype=np.complex128)

Z = np.array([
    [1, 0],
    [0, -1],
], dtype=np.complex128)

ZZ = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
],dtype=np.complex128).reshape(2, 2, 2, 2)

X = np.array([
    [0, 1],
    [1, 0],
], dtype=np.complex128)

def _make_zz_gate(ztime: float) -> NDArray:
    return np.cos(ztime) * np.eye(4, dtype=np.complex128).reshape(2, 2, 2, 2) - 1j * np.sin(ztime) * ZZ

def _make_z_gate(ztime: float) -> NDArray:
    return np.cos(ztime) * np.eye(2, dtype=np.complex128) - 1j * np.sin(ztime) * Z

def _make_x_gate(xtime: float) -> NDArray:
    return np.cos(xtime) * np.eye(2, dtype=np.complex128) - 1j * np.sin(xtime) * X

def _initialize_state_sv(context: Context) -> QuantumState:
    state = QuantumState(context.nodes_number)
    for pos in range(context.nodes_number):
        state.apply1(pos, FLIPPED_HADAMARD)
    return state

def _run_layer_sv(context: Context, xtime: float, ztime: float, state: QuantumState) -> None:
    for pos, ampl in context.node_to_ampl.items():
        z_gate = _make_z_gate(ztime * ampl)
        state.apply1(pos, z_gate)
    seen = set()
    for (lhs, rhs), ampl in context.edge_to_ampl.items():
        if (rhs, lhs) not in seen:
            seen.add((lhs, rhs))
            zz_gate = _make_zz_gate(ztime * ampl)
            state.apply2(lhs, rhs, zz_gate)
    x_gate = _make_x_gate(xtime)
    for pos in range(context.nodes_number):
        state.apply1(pos, x_gate)

# in the given API it is problematic to fix a seed, so we do not do this and the computation is not reproducable
def _measure_sv(context: Context, state: QuantumState) -> list:
    return [1 - 2 * state.measure(pos, uniform(0., 1., 1)) for pos in range(context.nodes_number)]

def _get_density_matrices_sv(context: Context, state: QuantumState) -> NDArray:
    return np.concatenate([state.dens1(pos)[np.newaxis] for pos in range(context.nodes_number)], axis=0)

def run_qa_exact(config) -> list:
    context = config_to_context(config)
    state = _initialize_state_sv(context)

    def execute_instruction(
            instruction: str | dict,
    ) -> None | list | NDArray:
        if isinstance(instruction, dict):
            return _run_layer_sv(context, instruction["xtime"], instruction["ztime"], state)
        elif instruction == "measure":
            return _measure_sv(context, state)
        elif instruction == "get_density_matrices":
            return _get_density_matrices_sv(context, state)
        else:
            raise ValueError(f"Unknown instruction {instruction}")
    instr_exec_iter = (execute_instruction(instr) for instr in context.instructions)
    return list(filter(lambda x: x is not None, instr_exec_iter))
