from functools import reduce
from math import cos, prod, sin, sqrt
from typing import Iterable, Optional, Self, Sequence, TypeVar, Generic, Callable
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from numpy.linalg import svd
from bqa.utils import NP_DTYPE


SQRT_NEG_1J = sqrt(2.) / 2. - 1j * sqrt(2.) / 2.


RawTensor = TypeVar("RawTensor")


class Tensor(ABC, Generic[RawTensor]):

    # these methods must be overloaded

    @classmethod
    @abstractmethod
    def make_from_list(cls, lst: list) -> Self:
        pass

    @classmethod
    @abstractmethod
    def make_from_numpy(cls, arr: NDArray) -> Self:
        pass

    @classmethod
    @abstractmethod
    def make_from_raw_tensor(cls, raw_tensor: RawTensor) -> Self:
        pass

    @staticmethod
    @abstractmethod
    def make_empty_raw_tensor(batch_size: int, shape: tuple[int, ...]) -> RawTensor:
        pass

    @property
    @abstractmethod
    def raw_tensor(self) -> RawTensor:
        pass

    @property
    @abstractmethod
    def numpy(self) -> NDArray:
        pass

    @staticmethod
    @abstractmethod
    def batch_gather_raw_tensor(raw_tensor: RawTensor, indices: RawTensor) -> RawTensor:
        pass

    @staticmethod
    @abstractmethod
    def get_batched_raw_tensor_l2_norm(raw_tensor: RawTensor) -> RawTensor:
        pass

    @staticmethod
    @abstractmethod
    def get_raw_tensor_l2_norm(raw_tensor: RawTensor) -> float:
        pass

    @staticmethod
    @abstractmethod
    def get_batched_raw_tensor_trace(raw_tensor: RawTensor) -> RawTensor:
        pass

    @staticmethod
    @abstractmethod
    def concatenate_raw_tensors(lhs: RawTensor, rhs: RawTensor, axis: int) -> RawTensor:
        pass

    @staticmethod
    @abstractmethod
    def get_raw_tensor_shape(raw_tensor: RawTensor) -> tuple[int, ...]:
        pass

    @staticmethod
    @abstractmethod
    def inverse_raw_tensor_elementwise(raw_tensor: RawTensor) -> RawTensor:
        pass

    @staticmethod
    @abstractmethod
    def batch_reshape_raw_tensor(
        raw_tensor: RawTensor,
        shape: tuple[int, ...],
    ) -> RawTensor:
        pass

    @staticmethod
    @abstractmethod
    def mul_by_float(
        raw_tensor: RawTensor,
        val: float | complex | int,
    ) -> RawTensor:
        pass

    @staticmethod
    @abstractmethod
    def batch_transpose_raw_tensor(
        raw_tensor: RawTensor, index_order: tuple[int, ...]
    ) -> RawTensor:
        pass

    @staticmethod
    @abstractmethod
    def turn_raw_tensor_last_index_into_diag_matrix(raw_tensor: RawTensor) -> RawTensor:
        pass

    @staticmethod
    @abstractmethod
    def complex_conj_raw_tensor(raw_tensor: RawTensor) -> RawTensor:
        pass

    @staticmethod
    @abstractmethod
    def batch_matmul_raw_tensor(
        lhs_raw_tensor: RawTensor, rhs_raw_tensor: RawTensor
    ) -> RawTensor:
        pass

    @staticmethod
    @abstractmethod
    def apply_x_to_phys_dim_raw(
        raw_tensor: RawTensor
    ) -> RawTensor:
        pass

    @staticmethod
    @abstractmethod
    def apply_z_to_phys_dim_raw(
        raw_tensor: RawTensor,
    ) -> RawTensor:
        pass

    @staticmethod
    @abstractmethod
    def assign_at_batch_indices_raw(
        dst_raw_tensor: RawTensor, src_raw_tensor: RawTensor, indices: RawTensor
    ) -> RawTensor:
        pass

    @staticmethod
    @abstractmethod
    def take_raw_tensor_elementwise_sqrt(raw_tensor: RawTensor) -> RawTensor:
        pass

    @staticmethod
    @abstractmethod
    def take_raw_tensor_batch_slice(
        raw_tensor: RawTensor, start: int, end: int
    ) -> RawTensor:
        pass

    @staticmethod
    @abstractmethod
    def broadcasted_mul_raw_tensors(lhs: RawTensor, rhs: RawTensor) -> RawTensor:
        pass

    @staticmethod
    @abstractmethod
    def broadcasted_sum_raw_tensors(lhs: RawTensor, rhs: RawTensor) -> RawTensor:
        pass

    @staticmethod
    @abstractmethod
    def broadcasted_sub_raw_tensors(lhs: RawTensor, rhs: RawTensor) -> RawTensor:
        pass

    @staticmethod
    @abstractmethod
    def get_batch_raw_tensor_svd_in_subspace(
        raw_tensor: RawTensor,
        pinv_eps: float,
    ) -> tuple[RawTensor, RawTensor, RawTensor]:
        pass

    @staticmethod
    @abstractmethod
    def get_raw_tensor_elementwise_pinv(raw_tensor: RawTensor) -> RawTensor:
        pass

    @staticmethod
    @abstractmethod
    def truncate_raw_tensor(raw_tensor: RawTensor, dims: Sequence[int]) -> RawTensor:
        pass

    @staticmethod
    @abstractmethod
    def measure_raw_tensor_by_position_in_place(
        raw_tensor: RawTensor, position: int, outcome: int
    ) -> None:
        pass

    @staticmethod
    @abstractmethod
    def get_sin_raw_tensor(raw_tensor: RawTensor) -> RawTensor:
        pass

    @staticmethod
    @abstractmethod
    def get_cos_raw_tensor(raw_tensor: RawTensor) -> RawTensor:
        pass

    @staticmethod
    @abstractmethod
    def compute_minimal_rank_from_raw_lmbds(lmbds: RawTensor, eps: float) -> int:
        pass

    # rest are implementations in terms of abstract methods (do not overload)

    @classmethod
    def make_from_iter(cls, it: Iterable) -> Self:
        return cls.make_from_list(list(it))

    @classmethod
    def make_empty(cls, batch_size: int, shape: tuple[int, ...]) -> Self:
        return cls.make_from_raw_tensor(cls.make_empty_raw_tensor(batch_size, shape))

    def apply_to_raw_tensor(self, func: Callable, *extra_args):
        return self.make_from_raw_tensor(func(self.raw_tensor, *extra_args))

    def assign_at_batch_indices(self, src: Self, indices: Self) -> Self:
        return self.apply_to_raw_tensor(
            self.assign_at_batch_indices_raw, src.raw_tensor, indices.raw_tensor
        )

    @property
    def raw_shape(self) -> tuple[int, ...]:
        return self.get_raw_tensor_shape(self.raw_tensor)

    def batch_slice(self, indices: Self) -> Self:
        assert len(indices.batch_shape) == 0, indices.batch_shape
        return self.apply_to_raw_tensor(
            self.batch_gather_raw_tensor, indices.raw_tensor
        )

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.raw_shape[1:]

    @property
    def batch_rank(self) -> int:
        return len(self.batch_shape)

    @property
    def batch_size(self) -> int:
        return self.raw_shape[0]

    def batch_truncate_all_but(self, dim: int, skip: Optional[list[int]] = None) -> Self:
        if skip is None:
            batch_dims = (min(d, dim) for d in self.batch_shape)
        else:
            batch_dims = (d if axis in skip else min(d, dim) for axis, d in enumerate(self.batch_shape))
        dims = (self.batch_size, *batch_dims)
        return self.apply_to_raw_tensor(self.truncate_raw_tensor, dims)

    def conj(self) -> Self:
        return self.apply_to_raw_tensor(self.complex_conj_raw_tensor)

    def measure(self, position: int, outcome: int) -> None:
        self.measure_raw_tensor_by_position_in_place(self.raw_tensor, position, outcome)

    def batch_normalize(self) -> Self:
        inv_norm = self.apply_to_raw_tensor(lambda raw_tensor: self.inverse_raw_tensor_elementwise(self.get_batched_raw_tensor_l2_norm(raw_tensor)))
        return self._mul_by_constants(inv_norm)

    def batch_trace_normalize(self) -> Self:
        inv_trace = self.apply_to_raw_tensor(lambda raw_tensor: self.inverse_raw_tensor_elementwise(self.get_batched_raw_tensor_trace(raw_tensor)))
        return self._mul_by_constants(inv_trace)

    def batch_reshape(self, shape: tuple[int, ...]) -> Self:
        return self.apply_to_raw_tensor(self.batch_reshape_raw_tensor, shape)

    def batch_transpose(self, indices_order: tuple[int, ...]) -> Self:
        return self.apply_to_raw_tensor(self.batch_transpose_raw_tensor, indices_order)

    def batch_diag(self) -> Self:
        return self.apply_to_raw_tensor(
            self.turn_raw_tensor_last_index_into_diag_matrix
        )

    def batch_matmul(self, other: Self) -> Self:
        return self.apply_to_raw_tensor(self.batch_matmul_raw_tensor, other.raw_tensor)

    def _batch_tensordot_by_axes_number(self, other: Self, axes_num: int) -> Self:
        assert axes_num > 0
        assert self.batch_shape[-axes_num:] == other.batch_shape[:axes_num]
        lhs_batch_shape = self.batch_shape
        rhs_batch_shape = other.batch_shape
        assert len(lhs_batch_shape) >= axes_num
        assert len(rhs_batch_shape) >= axes_num
        contraction_dim = prod(lhs_batch_shape[-axes_num:])
        assert contraction_dim == prod(rhs_batch_shape[:axes_num])
        new_batch_shape = (*lhs_batch_shape[:-axes_num], *rhs_batch_shape[axes_num:])
        lhs_mat = self.batch_reshape((-1, contraction_dim))
        rhs_mat = other.batch_reshape((contraction_dim, -1))
        return lhs_mat.batch_matmul(rhs_mat).batch_reshape(new_batch_shape)

    def _batch_tensordot(self, other: Self, axes: list[list[int]]) -> Self:
        lhs_axes, rhs_axes = axes
        axes_num = len(lhs_axes)
        assert len(rhs_axes) == axes_num
        assert len(set(lhs_axes)) == axes_num
        assert len(set(rhs_axes)) == axes_num
        assert all((a >= 0 and a < self.batch_rank for a in lhs_axes))
        assert all((a >= 0 and a < other.batch_rank for a in rhs_axes))
        lhs_new_index_order = (
            *(x for x in range(self.batch_rank) if x not in lhs_axes),
            *lhs_axes,
        )
        rhs_new_index_order = (
            *rhs_axes,
            *(x for x in range(other.batch_rank) if x not in rhs_axes),
        )
        lhs = self.batch_transpose(lhs_new_index_order)
        rhs = other.batch_transpose(rhs_new_index_order)
        return lhs._batch_tensordot_by_axes_number(rhs, axes_num)

    def batch_tensordot(self, other: Self, axes: list[list[int]] | int) -> Self:
        if isinstance(axes, int):
            return self._batch_tensordot_by_axes_number(other, axes)
        elif isinstance(axes, list):
            lhs_axes, rhs_axes = axes
            lhs_rank = self.batch_rank
            rhs_rank = other.batch_rank
            desug_axes = [
                [lhs_rank + x if x < 0 else x for x in lhs_axes],
                [rhs_rank + x if x < 0 else x for x in rhs_axes],
            ]
            return self._batch_tensordot(other, desug_axes)

    def _move_first_bond_to_last_pos(self) -> Self:
        batch_rank = self.batch_rank
        new_indices_order = (0, *range(2, batch_rank), 1)
        return self.batch_transpose(new_indices_order)

    def _apply_msgs_but_one(self, msgs: tuple[Self, ...], but: int) -> Self:

        def apply_msg(tensor: Self, pos_msg: tuple[int, Self]) -> Self:
            pos, msg = pos_msg
            return (
                tensor.batch_tensordot(msg, [[1], [1]])
                if pos != but
                else tensor._move_first_bond_to_last_pos()
            )

        return reduce(apply_msg, enumerate(msgs), self)

    def _compute_msg(self, self_conj: Self,  msgs: tuple[Self, ...], idx: int) -> Self:
        tensor_msgs = self._apply_msgs_but_one(msgs, idx)
        rank = self.batch_rank
        indices = [i for i in range(rank) if i != idx + 1]
        return (
            self_conj
            .batch_tensordot(tensor_msgs, [indices, indices])
            .batch_trace_normalize()
        )

    def pass_msgs(self, msgs: tuple[Self, ...]) -> tuple[Self, ...]:
        self_conj = self.conj()
        return tuple(self._compute_msg(self_conj, msgs, idx) for idx in range(len(msgs)))

    def apply_canonicalizers(self, canonicalizers: tuple[Self, ...]) -> Self:
        assert self.batch_rank - 1 == len(canonicalizers)
        return reduce(
            lambda t, c: t.batch_tensordot(c, [[1], [0]]), canonicalizers, self
        )

    def _apply_x_to_phys_dim(self) -> Self:
        return self.apply_to_raw_tensor(self.apply_x_to_phys_dim_raw)

    def _apply_z_to_phys_dim(self) -> Self:
        return self.apply_to_raw_tensor(self.apply_z_to_phys_dim_raw)

    def get_density_matrices(self, msgs: tuple[Self, ...]) -> Self:
        tensor_msgs = reduce(
            lambda tensor, msg: tensor.batch_tensordot(msg, [[1], [1]]), msgs, self
        )
        conj_tensor = self.conj()
        indices_to_contract = list(range(1, tensor_msgs.batch_rank))
        return tensor_msgs.batch_tensordot(
            conj_tensor, [indices_to_contract, indices_to_contract]
        ).batch_trace_normalize()

    def mul_by_lmbds(self, lmbds: tuple[Self, ...]) -> Self:
        lmbds_num = len(lmbds)
        batch_shape = self.batch_shape
        assert lmbds_num == self.batch_rank - 1

        def mul_by_lmbd(tensor: Self, pos_and_lmbd: tuple[int, Self]) -> Self:
            pos, lmbd = pos_and_lmbd
            dim = batch_shape[pos + 1]
            assert dim == lmbd.batch_shape[0]
            lmbd_new_shape = (1, *(dim if p == pos else 1 for p in range(lmbds_num)))
            return tensor * lmbd.batch_reshape(lmbd_new_shape)

        return reduce(mul_by_lmbd, enumerate(lmbds), self).batch_normalize()

    def sqrt(self) -> Self:
        return self.apply_to_raw_tensor(self.take_raw_tensor_elementwise_sqrt)

    def inv(self) -> Self:
        return self.apply_to_raw_tensor(self.inverse_raw_tensor_elementwise)

    def pinv(self) -> Self:
        return self.apply_to_raw_tensor(self.get_raw_tensor_elementwise_pinv)

    def get_batch_slice(self, indices: range) -> Self:
        assert indices.step == 1
        return self.apply_to_raw_tensor(
            self.take_raw_tensor_batch_slice, indices.start, indices.stop
        )

    def get_batch_svd(self, pinv_eps: float) -> tuple[Self, Self, Self]:
        u, s, vh = self.get_batch_raw_tensor_svd_in_subspace(self.raw_tensor, pinv_eps)
        return self.make_from_raw_tensor(u), self.make_from_raw_tensor(s), self.make_from_raw_tensor(vh)

    def decompose_iden_using_msgs(self, eps: float) -> tuple[Self, Self]:
        u, lmbd, uh = self.get_batch_svd(eps)
        lmbd_size = lmbd.batch_shape[0]
        lmbd_sqrt = lmbd.sqrt()
        lu = lmbd_sqrt.batch_reshape((lmbd_size, 1)) * uh
        lmbd_sqrt_pinv = lmbd_sqrt.pinv()
        ul = u * lmbd_sqrt_pinv.batch_reshape((1, lmbd_size))
        return ul, lu

    def get_dist(self, other: Self) -> float:
        return self.get_raw_tensor_l2_norm((self - other).raw_tensor) / self.get_raw_tensor_l2_norm((self + other).raw_tensor)

    def batch_concat(self, other: Self) -> Self:
        return self.apply_to_raw_tensor(self.concatenate_raw_tensors, other.raw_tensor, 0)

    def sin(self) -> Self:
        return self.apply_to_raw_tensor(self.get_sin_raw_tensor)

    def cos(self) -> Self:
        return self.apply_to_raw_tensor(self.get_cos_raw_tensor)

    def apply_x_gates(self, xtime: float) -> Self:
        return cos(xtime) * self - 1j * sin(xtime) * self._apply_x_to_phys_dim()

    def apply_z_gates(self, ztimes: Self) -> Self:
        return self._mul_by_constants(ztimes.cos()) - 1j * self._apply_z_to_phys_dim()._mul_by_constants(ztimes.sin())

    def _mul_by_constants(self, constants: Self) -> Self:
        rank = self.batch_rank
        return self * constants.batch_reshape(rank * (1,))

    def _batch_concatenate(self, other: Self, axis: int) -> Self:
        return self.apply_to_raw_tensor(self.concatenate_raw_tensors, other.raw_tensor, axis + 1)

    def _apply_conditional_z_gate_to_single_axis(self, axis: int, coupling: Self) -> Self:
        up = self._mul_by_constants(coupling.cos().sqrt())
        down = self._apply_z_to_phys_dim()._mul_by_constants(SQRT_NEG_1J * coupling.sin().sqrt())
        return up._batch_concatenate(down, axis)

    def apply_conditional_z_gates(self, couplings: Iterable[Self]) -> Self:
        def reduction_func(acc: Self, axis_and_coupling: tuple[int, Self]) -> Self:
            axis, coupling = axis_and_coupling
            return acc._apply_conditional_z_gate_to_single_axis(axis, coupling)
        return reduce(reduction_func, ((i + 1, c) for i, c in enumerate(couplings)), self).batch_normalize()

    def extend_msgs(self) -> Self:
        d, d_rhs = self.batch_shape
        assert d == d_rhs
        eye = self.make_from_numpy(np.eye(2, dtype=NP_DTYPE).reshape((1, 2, 2, 1, 1)))
        return (self.batch_reshape((1, 1, d, d)) * eye).batch_reshape((2 * d, 2 * d))

    def compute_minimal_rank_from_lmbd(self, eps: float) -> int:
        return self.compute_minimal_rank_from_raw_lmbds(self.raw_tensor, eps)

    def __str__(self):
        return f"{self.__class__.__name__}({self.raw_tensor.__str__()})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.raw_tensor.__repr__()})"

    def __mul__(self, other: Self | complex | float | int) -> Self:
        if isinstance(other, (complex, float, int)):
            return self.apply_to_raw_tensor(
                self.mul_by_float, other
            )
        elif isinstance(other, type(self)):
            return self.apply_to_raw_tensor(
                self.broadcasted_mul_raw_tensors, other.raw_tensor
            )
        else:
            return NotImplemented

    def __rmul__(self, other: complex | float | int) -> Self:
        return self * other

    def __add__(self, other: Self) -> Self:
        return self.apply_to_raw_tensor(
            self.broadcasted_sum_raw_tensors, other.raw_tensor
        )

    def __sub__(self, other: Self) -> Self:
        return self.apply_to_raw_tensor(
            self.broadcasted_sub_raw_tensors, other.raw_tensor
        )


class NumPyBackend(Tensor):

    def __init__(self, tensor: NDArray):
        self._tensor = tensor

    @classmethod
    def make_from_list(cls, lst: list) -> Self:
        dtype = np.intp if lst and isinstance(lst[0], int) else NP_DTYPE
        return cls(np.array(lst, dtype=dtype))

    @classmethod
    def make_from_numpy(cls, arr: NDArray) -> Self:
        return cls(arr)

    @classmethod
    def make_from_raw_tensor(cls, raw_tensor: NDArray) -> Self:
        return cls(raw_tensor)

    @property
    def raw_tensor(self) -> NDArray:
        return self._tensor

    @property
    def numpy(self) -> NDArray:
        return self._tensor

    @staticmethod
    def batch_gather_raw_tensor(raw_tensor: NDArray, indices: NDArray) -> NDArray:
        return raw_tensor[indices]

    @staticmethod
    def get_batched_raw_tensor_l2_norm(raw_tensor: NDArray) -> NDArray:
        shape = raw_tensor.shape
        rank = len(shape) - 1
        batch_size = shape[0]
        assert rank >= 0, shape
        raw_tensor_reshaped = raw_tensor.reshape((batch_size, -1))
        norms = np.linalg.norm(raw_tensor_reshaped, axis=1)
        return norms

    @staticmethod
    def get_raw_tensor_l2_norm(raw_tensor: NDArray) -> float:
        return float(np.linalg.norm(raw_tensor))

    @staticmethod
    def get_batched_raw_tensor_trace(raw_tensor: NDArray) -> NDArray:
        return np.trace(raw_tensor, axis1=-2, axis2=-1)

    @staticmethod
    def apply_x_to_phys_dim_raw(raw_tensor: NDArray) -> NDArray:
        return raw_tensor[:, ::-1]

    @staticmethod
    def apply_z_to_phys_dim_raw(raw_tensor: NDArray) -> NDArray:
        new_raw_tensor = raw_tensor.copy()
        new_raw_tensor[:, 1] *= -1.
        return  new_raw_tensor

    @staticmethod
    def make_empty_raw_tensor(batch_size: int, shape: tuple[int, ...]) -> NDArray:
        return np.empty((batch_size, *shape), NP_DTYPE)

    @staticmethod
    def assign_at_batch_indices_raw(
        dst_raw_tensor: NDArray, src_raw_tensor: NDArray, indices: NDArray
    ) -> NDArray:
        dst_raw_tensor[indices] = src_raw_tensor
        return dst_raw_tensor

    @staticmethod
    def get_raw_tensor_shape(raw_tensor: NDArray) -> tuple[int, ...]:
        return raw_tensor.shape

    @staticmethod
    def inverse_raw_tensor_elementwise(raw_tensor: NDArray) -> NDArray:
        return 1 / raw_tensor

    @staticmethod
    def batch_reshape_raw_tensor(
        raw_tensor: NDArray, shape: tuple[int, ...]
    ) -> NDArray:
        batch_size = raw_tensor.shape[0]
        return raw_tensor.reshape((batch_size, *shape))

    @staticmethod
    def turn_raw_tensor_last_index_into_diag_matrix(raw_tensor: NDArray) -> NDArray:
        last_index_dim = raw_tensor.shape[-1]
        return raw_tensor[..., np.newaxis] * np.eye(last_index_dim)

    @staticmethod
    def complex_conj_raw_tensor(raw_tensor: NDArray) -> NDArray:
        return raw_tensor.conj()

    @staticmethod
    def batch_matmul_raw_tensor(
        lhs_raw_tensor: NDArray, rhs_raw_tensor: NDArray
    ) -> NDArray:
        return lhs_raw_tensor @ rhs_raw_tensor

    @staticmethod
    def batch_transpose_raw_tensor(
        raw_tensor: NDArray, index_order: tuple[int, ...]
    ) -> NDArray:
        index_order_shifted = (0, *(i + 1 for i in index_order))
        return raw_tensor.transpose(index_order_shifted)

    @staticmethod
    def take_raw_tensor_elementwise_sqrt(raw_tensor: NDArray) -> NDArray:
        return np.sqrt(raw_tensor)

    @staticmethod
    def take_raw_tensor_batch_slice(
        raw_tensor: NDArray, start: int, end: int
    ) -> NDArray:
        return raw_tensor[start:end]

    @staticmethod
    def broadcasted_mul_raw_tensors(lhs: NDArray, rhs: NDArray) -> NDArray:
        return lhs * rhs

    @staticmethod
    def broadcasted_sum_raw_tensors(lhs: NDArray, rhs: NDArray) -> NDArray:
        return lhs + rhs

    @staticmethod
    def broadcasted_sub_raw_tensors(lhs: NDArray, rhs: NDArray) -> NDArray:
        return lhs - rhs

    @staticmethod
    def get_batch_raw_tensor_svd_in_subspace(
        raw_tensor: NDArray,
        pinv_eps: float,
    ) -> tuple[NDArray, NDArray, NDArray]:
        u, s, vh = svd(raw_tensor, full_matrices=False)
        mask = s > pinv_eps
        s = (s * mask).astype(NP_DTYPE)
        return u * mask[..., np.newaxis, :], s.astype(NP_DTYPE), vh * mask[..., np.newaxis]

    @staticmethod
    def get_raw_tensor_elementwise_pinv(raw_tensor: NDArray) -> NDArray:
        dtype = raw_tensor.dtype
        return np.divide(
            1.0,
            raw_tensor,
            out=np.zeros_like(raw_tensor, dtype),
            where=raw_tensor > np.finfo(NP_DTYPE).eps,
        )

    @staticmethod
    def measure_raw_tensor_by_position_in_place(
        raw_tensor: NDArray, position: int, outcome
    ) -> None:
        raw_tensor[position, 0 if outcome else 1] = 0.0
        raw_tensor[position, 1 if outcome else 0] = 1.0

    @staticmethod
    def truncate_raw_tensor(raw_tensor: NDArray, dims: Sequence[int]) -> NDArray:
        return raw_tensor[tuple(slice(0, d) for d in dims)]

    @staticmethod
    def mul_by_float(raw_tensor: NDArray, val: float | complex | int) -> NDArray:
        return raw_tensor * val

    @staticmethod
    def get_sin_raw_tensor(raw_tensor: NDArray) -> NDArray:
        return np.sin(raw_tensor)

    @staticmethod
    def get_cos_raw_tensor(raw_tensor: NDArray) -> NDArray:
        return np.cos(raw_tensor)

    @staticmethod
    def concatenate_raw_tensors(lhs: NDArray, rhs: NDArray, axis: int) -> NDArray:
        return np.concatenate([lhs, rhs], axis)

    @staticmethod
    def compute_minimal_rank_from_raw_lmbds(lmbds: NDArray, eps: float) -> int:
        return int(np.max((lmbds > eps).sum(-1)))
