from os import environ
from functools import cache
from typing import TypeVar
from numpy import complex64, complex128

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


NP_DTYPE = dispatch_precision(complex64, complex128)
