from functools import reduce
from itertools import chain
from operator import mul
from typing import Iterable, Self, TypeVar, Generic, Callable
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

RawTensor = TypeVar("RawTensor")

class Tensor(ABC, Generic[RawTensor]):

    # these methods must be overloaded

    @classmethod
    @abstractmethod
    def from_list(cls, lst: list) -> Self:
        pass

    @classmethod
    @abstractmethod
    def from_numpy(cls, arr: NDArray) -> Self:
        pass

    @classmethod
    @abstractmethod
    def from_raw_tensor(cls, raw_tensor: RawTensor) -> Self:
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
    def get_batched_raw_tensor_trace(raw_tensor: RawTensor) -> RawTensor:
        pass

    @staticmethod
    @abstractmethod
    def batch_multiply_raw_tensor_by_constant(raw_tensor: RawTensor, constant: RawTensor) -> RawTensor:
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
    def batch_reshape_raw_tensor(raw_tensor: RawTensor, shape: tuple[int, ...]) -> RawTensor:
        pass

    @staticmethod
    @abstractmethod
    def batch_transpose_raw_tensor(raw_tensor: RawTensor, index_order: tuple[int, ...]) -> RawTensor:
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
    def batch_matmul_raw_tensor(lhs_raw_tensor: RawTensor, rhs_raw_tensor: RawTensor) -> RawTensor:
        pass

    # rest are implementations in terms of abstract methods (do not overload)

    @classmethod
    def from_iter(cls, it: Iterable) -> Self:
        return cls.from_list(list(it))

    def apply_to_raw_tensor(self, func: Callable, *extra_args):
        return self.from_raw_tensor(func(self.raw_tensor, *extra_args))

    @property
    def raw_shape(self) -> tuple[int, ...]:
        return self.get_raw_tensor_shape(self.raw_tensor)

    def batch_slice(self, indices: Self) -> Self:
        assert len(indices.batch_shape) == 0, indices.batch_shape
        return self.apply_to_raw_tensor(self.batch_gather_raw_tensor, indices.raw_tensor)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.raw_shape[1:]

    @property
    def batch_rank(self) -> int:
        return len(self.batch_shape)

    @property
    def batch_size(self) -> int:
        return self.raw_shape[0]

    def conj(self) -> Self:
        return self.apply_to_raw_tensor(self.complex_conj_raw_tensor)

    def batch_normalize(self) -> Self:
        raw_tensor = self.raw_tensor
        inv_norm = self.inverse_raw_tensor_elementwise(self.get_batched_raw_tensor_l2_norm(raw_tensor))
        return self.apply_to_raw_tensor(self.batch_multiply_raw_tensor_by_constant, inv_norm)

    def batch_trace_normalize(self) -> Self:
        raw_tensor = self.raw_tensor
        inv_trace = self.inverse_raw_tensor_elementwise(self.get_batched_raw_tensor_trace(raw_tensor))
        return self.apply_to_raw_tensor(self.batch_multiply_raw_tensor_by_constant, inv_trace)

    def batch_reshape(self, shape: tuple[int, ...]) -> Self:
        return self.apply_to_raw_tensor(self.batch_reshape_raw_tensor, shape)

    def batch_transpose(self, indices_order: tuple[int, ...]) -> Self:
        return self.apply_to_raw_tensor(self.batch_transpose_raw_tensor, indices_order)

    def batch_diag(self) -> Self:
        return self.apply_to_raw_tensor(self.turn_raw_tensor_last_index_into_diag_matrix)

    def batch_matmul(self, other: Self) -> Self:
        return self.apply_to_raw_tensor(self.batch_matmul_raw_tensor, other.raw_tensor)

    def _batch_tensordot(self, other: Self, axes: list[list[int]]) -> Self:
        lhs_axes, rhs_axes = axes
        contraction_dim = reduce(mul, map(lambda pair: pair[1], filter(lambda pair: pair[0] in lhs_axes, enumerate(self.batch_shape))))
        new_shape = (*map(lambda pair: pair[1], filter(lambda pair: pair[0] not in lhs_axes, enumerate(self.batch_shape))),
                     *map(lambda pair: pair[1], filter(lambda pair: pair[0] not in rhs_axes, enumerate(other.batch_shape))))
        lhs_initial_index_order = range(self.batch_rank)
        rhs_initial_index_order = range(other.batch_rank)
        lhs_new_index_order = tuple(chain(filter(lambda x: x not in lhs_axes, lhs_initial_index_order), lhs_axes))
        rhs_new_index_order = tuple(chain(rhs_axes, filter(lambda x: x not in rhs_axes, rhs_initial_index_order)))
        lhs_mat = self.batch_transpose(lhs_new_index_order).batch_reshape((-1, contraction_dim))
        rhs_mat = other.batch_transpose(rhs_new_index_order).batch_reshape((contraction_dim, -1))
        return lhs_mat.batch_matmul(rhs_mat).batch_reshape(new_shape)

    def batch_tensordot(self, other: Self, axes: list[list[int]] | int) -> Self:
        if isinstance(axes, int):
            return self.batch_tensordot(other, [list(range(-axes, 0)), list(range(axes))])
        elif isinstance(axes, list):
            lhs_axes, rhs_axes = axes
            lhs_rank = self.batch_rank
            rhs_rank = other.batch_rank
            desug_axes = [list(map(lambda x: lhs_rank + x if x < 0 else x, lhs_axes)),
                          list(map(lambda x: rhs_rank + x if x < 0 else x, rhs_axes))]
            return self._batch_tensordot(other, desug_axes)

    def pass_msgs(self, msgs: tuple[Self, ...]) -> tuple[Self, ...]:
        raise NotImplementedError()

    def get_density_matrices(self, msgs: tuple[Self, ...]) -> Self:
        tensor_msgs = reduce(lambda tensor, msg: tensor.batch_tensordot(msg, [[1], [1]]), msgs, self)
        conj_tensor = self.conj()
        indices_to_contract = list(range(1, tensor_msgs.batch_rank))
        return tensor_msgs.batch_tensordot(conj_tensor, [indices_to_contract, indices_to_contract]).batch_trace_normalize()

    def __str__(self):
        return f"{self.__class__.__name__}({self.raw_tensor.__str__()})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.raw_tensor.__repr__()})"

class NumPyBackend(Tensor):

    def __init__(self, tensor: NDArray):
        self._tensor = tensor

    @classmethod
    def from_list(cls, lst: list) -> Self:
        return cls(np.array(lst))        

    @classmethod
    def from_numpy(cls, arr: NDArray) -> Self:
        return cls(arr)

    @classmethod
    def from_raw_tensor(cls, raw_tensor: NDArray) -> Self:
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
    def get_batched_raw_tensor_trace(raw_tensor: NDArray) -> NDArray:
        return np.trace(raw_tensor, axis1 = -2, axis2 = -1)

    @staticmethod
    def batch_multiply_raw_tensor_by_constant(raw_tensor: NDArray, constant: NDArray) -> NDArray:
        shape = raw_tensor.shape
        rank = len(shape) - 1
        return constant.reshape((-1, *(rank * (1,)))) * raw_tensor

    @staticmethod
    def get_raw_tensor_shape(raw_tensor: NDArray) -> tuple[int, ...]:
        return raw_tensor.shape

    @staticmethod
    def inverse_raw_tensor_elementwise(raw_tensor: NDArray) -> NDArray:
        return 1 / raw_tensor

    @staticmethod
    def batch_reshape_raw_tensor(raw_tensor: NDArray, shape: tuple[int, ...]) -> NDArray:
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
    def batch_matmul_raw_tensor(lhs_raw_tensor: NDArray, rhs_raw_tensor: NDArray) -> NDArray:
        return lhs_raw_tensor @ rhs_raw_tensor

    @staticmethod
    def batch_transpose_raw_tensor(raw_tensor: NDArray, index_order: tuple[int, ...]) -> NDArray:
        index_order_shifted = tuple(chain([0], map(lambda x: x + 1, index_order)))
        return raw_tensor.transpose(index_order_shifted)

