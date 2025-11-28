import logging
from typing import Any, Callable, Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")

S = TypeVar("S")


class ConfigSyntaxError(ValueError):
    pass


log = logging.getLogger(__name__)


def _filtermap(it: Iterable[T], func: Callable[[T], Optional[S]]) -> Iterator[S]:
    for elem in it:
        if (val := func(elem)) is not None:
            yield val


def _is_sequential(smthng) -> bool:
    return isinstance(smthng, (list, tuple))


def _ok_or_default_and_warn(value: Optional[Any], default: T, msg: str) -> Any | T:
    if value is None:
        log.warning(msg)
        return default
    else:
        return value


def _unwrap_or(value: Optional[T], err_msg: str) -> T:
    if value is None:
        raise ConfigSyntaxError(err_msg)
    else:
        return value


def _analyse_atomic_value(
    func: Callable[[Any], bool],
    value: T,
    err_msg: str,
) -> T:
    if func(value):
        return value
    else:
        raise ConfigSyntaxError(f"Invalid value {value}, {err_msg}")


def _analyse_non_neg_number(value) -> float:
    number = _analyse_atomic_value(
        lambda x: isinstance(x, (float, int)) and x >= 0.0,
        value,
        "must be a non-negative `float` or `int` number",
    )
    return float(number)


def _analyse_0_to_1_number(value) -> float:
    number = _analyse_atomic_value(
        lambda x: isinstance(x, (float, int)) and x >= 0.0 and x <= 1.0,
        value,
        "must be a `float` or `int` number from [0, 1]",
    )
    return float(number)

def _analyse_half_to_1_number(value) -> float:
    number = _analyse_atomic_value(
        lambda x: isinstance(x, (float, int)) and x >= 0.5 and x <= 1.0,
        value,
        "must be a `float` or `int` number from [0.5, 1]",
    )
    return float(number)


def _analyse_non_neg_int(value) -> int:
    return _analyse_atomic_value(
        lambda x: isinstance(x, int) and x >= 0, value, "must be a non-negative `int`"
    )


def _analyse_positive_int(value) -> int:
    return _analyse_atomic_value(
        lambda x: isinstance(x, int) and x > 0, value, "must be a positive `int`"
    )


def _analyse_number(value) -> float:
    number = _analyse_atomic_value(
        lambda x: isinstance(x, (float, int)),
        value,
        "must be a `float` or `int` number",
    )
    return float(number)
