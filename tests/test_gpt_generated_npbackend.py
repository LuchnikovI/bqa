import numpy as np
import pytest

from bqa.backends import NumPyBackend


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------


def rand_batch(batch=3, *shape, dtype=np.float64):
    return np.random.randn(batch, *shape).astype(dtype)


def make(b):
    return NumPyBackend(b)


# ---------------------------------------------------------
# 1. Basic construction
# ---------------------------------------------------------


def test_make_from_list_int():
    t = NumPyBackend.make_from_list([1, 2, 3])
    assert isinstance(t.raw_tensor, np.ndarray)
    assert t.raw_tensor.dtype == np.intp
    assert np.all(t.raw_tensor == np.array([1, 2, 3], dtype=np.intp))


def test_make_from_list_float():
    t = NumPyBackend.make_from_list([1.0, -2.0])
    assert t.raw_tensor.dtype == np.complex64


def test_make_from_numpy_identity():
    arr = np.arange(6).reshape(2, 3)
    t = NumPyBackend.make_from_numpy(arr)
    assert np.shares_memory(arr, t.raw_tensor) is True


# ---------------------------------------------------------
# 2. Basic elementwise ops: sqrt, relu, inverse, pinv
# ---------------------------------------------------------


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4)])
def test_elementwise_sqrt(shape):
    x = rand_batch(1, *shape)[0] ** 2
    t = make(x)
    out = t.sqrt().raw_tensor
    assert np.allclose(out, np.sqrt(x))


def test_relu():
    x = np.array([[[-1.0, +2.0]]])
    t = make(x)
    out = t.relu().raw_tensor
    assert np.allclose(out, np.array([[0.0, 2.0]]))


def test_inverse_no_zeros():
    x = rand_batch(3, 4, 5) + 1.0  # shift away from zero
    t = make(x)
    out = t.inv().raw_tensor
    assert np.allclose(out, 1 / x)


def test_pinv_small_threshold():
    x = np.array([[0.0, 0.1, 2.0]])
    t = make(x)
    out = t.pinv(1e-3).raw_tensor
    expected = np.array([0.0, 1 / 0.1, 1 / 2.0])
    assert np.allclose(out, expected)


# ---------------------------------------------------------
# 3. Shape semantics: batch, transpose, reshape
# ---------------------------------------------------------


def test_batch_shape_rank_size():
    x = rand_batch(5, 2, 3, 4)
    t = make(x)
    assert t.batch_size == 5
    assert t.batch_shape == (2, 3, 4)
    assert t.batch_rank == 3


def test_batch_reshape():
    x = rand_batch(4, 3, 2)
    t = make(x)
    out = t.batch_reshape((6,))  # collapse 3*2
    assert out.raw_tensor.shape == (4, 6)
    assert np.allclose(out.raw_tensor, x.reshape(4, 6))


def test_batch_transpose():
    x = rand_batch(3, 2, 4, 5)  # batch=3, dims=2,4,5
    t = make(x)
    out = t.batch_transpose((2, 0, 1))
    assert out.raw_tensor.shape == (3, 5, 2, 4)
    assert np.allclose(out.raw_tensor, np.transpose(x, (0, 3, 1, 2)))


# ---------------------------------------------------------
# 4. Batch matmul
# ---------------------------------------------------------


def test_batch_matmul():
    A = rand_batch(3, 4, 2)
    B = rand_batch(3, 2, 5)
    tA, tB = make(A), make(B)
    out = tA.batch_matmul(tB).raw_tensor
    assert np.allclose(out, A @ B)


# ---------------------------------------------------------
# 5. Tensordot tests
# ---------------------------------------------------------


def test_batch_tensordot_axes_int():
    A = rand_batch(4, 3, 2)
    B = rand_batch(4, 2, 5)
    tA, tB = make(A), make(B)
    out = tA.batch_tensordot(tB, axes=1).raw_tensor
    expected = np.matmul(A.reshape(4, 3, 2), B.reshape(4, 2, 5))
    assert np.allclose(out, expected)


def test_batch_tensordot_axes_list():
    A = rand_batch(2, 3, 4, 5)
    B = rand_batch(2, 5, 7, 3)
    tA, tB = make(A), make(B)
    out = tA.batch_tensordot(tB, axes=[[2, 0], [0, 2]]).raw_tensor
    expected = np.concatenate(
        list(
            map(
                lambda lhs, rhs: np.tensordot(lhs, rhs, axes=((2, 0), (0, 2)))[
                    np.newaxis
                ],
                A,
                B,
            )
        ),
        axis=0,
    )
    assert expected.shape == out.shape
    assert np.allclose(out, expected)


# ---------------------------------------------------------
# 6. diag expansion
# ---------------------------------------------------------


def test_diag_expand():
    x = rand_batch(4, 6)  # shape (4,6)
    t = make(x)
    out = t.batch_diag().raw_tensor  # expected shape (4,6,6)
    assert out.shape == (4, 6, 6)
    for b in range(4):
        assert np.allclose(np.diag(x[b]), out[b])


# ---------------------------------------------------------
# 7. batch gather, batch slice
# ---------------------------------------------------------


def test_batch_gather():
    x = rand_batch(6, 3, 3)
    t = make(x)
    idx = make(np.array([0, 2, 5]))
    out = t.batch_slice(idx).raw_tensor
    assert np.allclose(out, x[[0, 2, 5]])


def test_batch_slice_range():
    x = rand_batch(10, 3, 3)
    t = make(x)
    out = t.get_batch_slice(range(2, 7)).raw_tensor
    assert np.allclose(out, x[2:7])


# ---------------------------------------------------------
# 8. SVD tests
# ---------------------------------------------------------


def test_batch_svd_reconstruct():
    X = rand_batch(4, 5, 5)
    t = make(X)
    u, s, vh = t.get_batch_svd()
    U = u.raw_tensor
    S = s.raw_tensor
    VH = vh.raw_tensor

    rec = U @ np.diag(S[0]) @ VH[0]  # but need per batch
    # general reconstruction:
    rec = np.array([U[b] @ np.diag(S[b]) @ VH[b] for b in range(4)])

    assert np.allclose(rec, X, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------
# 9. broadcasted_mul_raw_tensors
# ---------------------------------------------------------


def test_broadcast_mul():
    A = rand_batch(5, 3, 2)
    B = rand_batch(5, 1, 2)
    tA, tB = make(A), make(B)
    out = (tA * tB).raw_tensor
    assert np.allclose(out, A * B)


# ---------------------------------------------------------
# 10. batch_multiply_by_constants correctness
# ---------------------------------------------------------


def test_batch_multiply_constants():
    X = rand_batch(4, 3, 3)
    c = np.array([2.0, 0.5, -1.0, 1.0])
    tX = make(X)
    out = tX.apply_to_raw_tensor(
        NumPyBackend.batch_multiply_raw_tensor_by_constants, c
    ).raw_tensor

    expected = np.array([c[b] * X[b] for b in range(4)])
    assert np.allclose(out, expected)


# ---------------------------------------------------------
# 11. pinv correctness for matrices
# ---------------------------------------------------------


def test_elementwise_pinv_behavior():
    X = np.array([[0.001, 0.0, -0.2]])
    t = make(X)
    out = t.pinv(1e-3).raw_tensor
    expected = np.array([[1000.0, 0.0, -5.0]])
    assert np.allclose(out, expected)


# ---------------------------------------------------------
# 12. conj
# ---------------------------------------------------------


def test_conj():
    Z = rand_batch(3, 2) + 1j * rand_batch(3, 2)
    t = make(Z)
    out = t.conj().raw_tensor
    assert np.allclose(out, np.conj(Z))


# ---------------------------------------------------------
# 13. smoke test: message passing logic doesn’t crash
# ---------------------------------------------------------


def test_msgs_sqrt_and_pinv_sqrt_smoke():
    # Use small PSD matrices
    A = rand_batch(3, 4, 4)
    A = A @ np.transpose(A, (0, 2, 1)) + np.eye(4)  # make PSD
    t = make(A)
    eps = 1e-6
    sqrt_msg, pinv_sqrt_msg = t.get_msgs_sqrt_and_pinv_sqrt(eps)
    assert sqrt_msg.raw_tensor.shape == A.shape
    assert pinv_sqrt_msg.raw_tensor.shape == A.shape
    # ensure they are finite
    assert np.isfinite(sqrt_msg.raw_tensor).all()
    assert np.isfinite(pinv_sqrt_msg.raw_tensor).all()
