from os import environ
from functools import lru_cache, reduce
from typing import TypeVar, Callable

T = TypeVar("T")
U = TypeVar("U")
K = TypeVar("K")


def dict_map(func: Callable[[K, T], U], dct: dict[K, T]) -> dict[K, U]:
    return {k : func(k, v) for k, v in dct.items()}

def dict_reduce(func: Callable[[U, K, T], U], dct: dict[K, T], init: U) -> U:
    return reduce(lambda acc, pair: func(acc, pair[0], pair[1]), dct.items(), init)

@lru_cache(None)
def dispatch_precision(single: T, double: T) -> T:
    precision = (environ.get("BQA_PRECISION") or "single").lower()
    if precision == "single":
        return single
    elif precision == "double":
        return double
    else:
        raise ValueError(f"Unknown value of BQA_PRECISION environment variable {precision}")
