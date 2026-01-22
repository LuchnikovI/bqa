from os import environ
from functools import cache
from typing import TypeVar
from numpy import complex64, complex128
from numpy.typing import NDArray

T = TypeVar("T")


@cache
def dispatch_precision(single: T, double: T) -> T:
    precision = (environ.get("BQA_PRECISION") or "double").lower()
    if precision == "single":
        return single
    elif precision == "double":
        return double
    else:
        raise ValueError(
            f"Unknown value of BQA_PRECISION environment variable {precision}"
        )


def convert_density_matrix_to_bloch_vector(density_matrix: NDArray) -> list[float]:
    x = float((density_matrix[0, 1] + density_matrix[1, 0]).real)
    y = float((density_matrix[1, 0] - density_matrix[0, 1]).imag)
    z = float((density_matrix[0, 0] - density_matrix[1, 1]).real)
    return [x, y, z]


NP_DTYPE = dispatch_precision(complex64, complex128)
