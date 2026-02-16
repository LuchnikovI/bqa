try:
    from itertools import chain
    import cupy as cp

    def get_letter(num: int) -> str:
        return chr(ord('a') + num + 1)

    def get_einsum_subscripts(lhs_rank: int, rhs_rank: int, axes: list[list[int]]) -> str:
        lhs_axes, rhs_axes = axes
        if lhs_rank + rhs_rank > 25:
            raise NotImplementedError(f"CuPy based batch tensordot is not implemented of total number of indices > 25, got twho tensors of rank {lhs_rank}, {rhs_rank}, that in total have {lhs_rank + rhs_rank} indices")
        assert len(lhs_axes) == len(rhs_axes)
        assert all(0 <= x < lhs_rank for x in lhs_axes)
        assert all(0 <= x < rhs_rank for x in rhs_axes)
        lhs_subscript = chain(
            'a',
            map(get_letter, range(lhs_rank)),
        )
        axis_to_code = {axis : code for axis, code in zip(rhs_axes, map(get_letter, lhs_axes))}
        rhs_subscript = chain(
            'a',
            (axis_to_code[axis] if axis in axis_to_code else get_letter(lhs_rank + axis) for axis in range(rhs_rank)),
        )
        result_subscript = chain(
            'a',
            (get_letter(axis) for axis in range(lhs_rank) if axis not in lhs_axes),
            (get_letter(axis + lhs_rank) for axis in range(rhs_rank) if axis not in rhs_axes),
        )
        full_subscripts = chain(
            lhs_subscript,
            ',',
            rhs_subscript,
            "->",
            result_subscript,
        )
        return "".join(full_subscripts)

    def batch_cutensordot(lhs_tensor, rhs_tensor, axes: list[list[int]] | int):
        lhs_rank = len(lhs_tensor.shape) - 1
        rhs_rank = len(rhs_tensor.shape) - 1
        assert lhs_rank >= 0
        assert rhs_rank >= 0
        if isinstance(axes, list) and len(axes) == 2:
            lhs_axes, rhs_axes = axes
            desug_axes = [
                [lhs_rank + a if a < 0 else a for a in lhs_axes],
                [rhs_rank + a if a < 0 else a for a in rhs_axes],
            ]
        elif isinstance(axes, int) and axes < lhs_rank and axes < rhs_rank:
            desug_axes = [
                list(range(lhs_rank - axes, lhs_rank)),
                list(range(axes)),
            ]
        else:
            raise AssertionError(f"Invalid axes {axes}")
        subscripts = get_einsum_subscripts(lhs_rank, rhs_rank, desug_axes)
        return cp.einsum(subscripts, lhs_tensor, rhs_tensor)

    def batch_cuapply_msgs(tensor, msg, idx: int):
        assert idx >= 0
        rank = len(tensor.shape) - 1
        assert idx < rank
        assert rank >= 0
        tensor_subscript = chain("a", map(get_letter, range(rank)))
        msg_subscript = ("a", get_letter(rank), get_letter(idx))
        result_subscript = chain("a", (get_letter(rank if a == idx else a) for a in range(rank)))
        full_subscripts = "".join(chain(
            tensor_subscript,
            ",",
            msg_subscript,
            "->",
            result_subscript,
        ))
        return cp.einsum(full_subscripts, tensor, idx)
        

except ImportError:
    def batch_cutensordot(lhs_tensor, rhs_tensor, axes: list[list[int]] | int):
        raise NotImplementedError("CuPy is missing, batch_cutensordot is not implemented")
