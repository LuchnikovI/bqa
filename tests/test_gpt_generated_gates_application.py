import numpy as np
import pytest
from math import pi
from bqa.backends import NumPyBackend
from bqa.utils import NP_DTYPE


def make_single_qubit_tensor(state):
    """
    state: array of shape (2,)
    returns Tensor with shape (batch=1, phys=2)
    """
    return NumPyBackend.make_from_numpy(
        np.asarray(state, dtype=NP_DTYPE)[None, :]
    )


def density_from_tensor(t):
    """
    For single-qubit tensor with no bonds.
    """
    psi = t.numpy[0]
    return np.outer(psi.conj(), psi)


def test_x_squared_identity():
    psi = make_single_qubit_tensor([1.0, 0.0])
    psi2 = psi._apply_x_to_phys_dim()._apply_x_to_phys_dim()
    assert np.allclose(psi2.numpy, psi.numpy)


def test_x_rotation_pi_global_phase():
    psi = make_single_qubit_tensor([1.0, 0.0])
    psi_rot = psi.apply_x_gates(pi)

    # should be -|0>
    assert np.allclose(psi_rot.numpy, -psi.numpy)


def test_x_eigenstate_minus():
    psi_minus = make_single_qubit_tensor(
        [1/np.sqrt(2), -1/np.sqrt(2)]
    )
    psi_rot = psi_minus.apply_x_gates(0.37)

    # density matrix must be invariant
    rho0 = density_from_tensor(psi_minus)
    rho1 = density_from_tensor(psi_rot)
    assert np.allclose(rho0, rho1, atol=1e-12)


def test_z_squared_identity():
    psi = make_single_qubit_tensor([0.4, 0.7])
    psi2 = psi._apply_z_to_phys_dim()._apply_z_to_phys_dim()
    assert np.allclose(psi2.numpy, psi.numpy)


def test_z_rotation_pi():
    psi = make_single_qubit_tensor([0.6, 0.8])
    ztime = pi
    ztimes = NumPyBackend.make_from_numpy(np.array([ztime], NP_DTYPE))
    psi_rot = psi.apply_z_gates(ztimes)

    assert np.allclose(psi_rot.numpy, -psi.numpy)


@pytest.mark.parametrize("angle", [0.1, 0.7, 1.3])
def test_single_qubit_unitarity(angle):
    psi = make_single_qubit_tensor([0.6, 0.8])
    psi = psi.apply_x_gates(angle)
    ztimes = NumPyBackend.make_from_numpy(np.array([angle], NP_DTYPE))
    psi = psi.apply_z_gates(ztimes)
    rho1 = density_from_tensor(psi)
    assert np.allclose(np.trace(rho1), 1.0, atol=1e-12)
    assert np.allclose(rho1, rho1.conj().T)

def test_conditional_z_doubles_bond():
    psi = make_single_qubit_tensor([[[1.0]], [[0.0]]])
    coupling = NumPyBackend.make_from_numpy(np.array([0.5], NP_DTYPE))
    psi2 = psi.apply_conditional_z_gates([coupling, coupling])
    assert psi2.batch_rank == 3
    assert psi2.batch_shape == (2, 2, 2)

def test_conditional_z_zero_is_identity():
    psi = make_single_qubit_tensor([[0.5], [0.8]]).batch_normalize()
    coupling = NumPyBackend.make_from_numpy(np.array([0.0], NP_DTYPE))
    psi2 = psi.apply_conditional_z_gates([coupling])
    recovered = psi2.numpy[:, :, 0]
    assert np.allclose(recovered, psi.numpy[:, :, 0])

def test_small_layer_does_not_break():
    psi = make_single_qubit_tensor([1/np.sqrt(2), 1/np.sqrt(2)])

    psi = psi.apply_x_gates(0.3)
    ztimes = NumPyBackend.make_from_numpy(np.array([0.4], NP_DTYPE))
    psi = psi.apply_z_gates(ztimes)

    rho = density_from_tensor(psi)

    assert np.allclose(np.trace(rho), 1.0)
    assert np.all(np.linalg.eigvals(rho).real >= -1e-6)
