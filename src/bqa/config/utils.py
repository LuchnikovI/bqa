import logging

class ConfigSyntaxError(ValueError):
    pass


log = logging.getLogger(__name__)


def _is_sequential(smthng):
    return isinstance(smthng, (list, tuple))


def _get_or_default_and_warn(dct, key, default):
    val = dct.get(key)
    if val is None:
        log.warning(f"`{key}` field is missing in {dct}, set to default {default}")
        val = default
    return val

def _get_or_raise(dct, key):
    val = dct.get(key)
    if val is None:
        raise ConfigSyntaxError(f"`{key}` field is missing in {dct}")
    return val


def _analyse_atomic_value(func, value, err_msg):
    if func(value):
        return value
    else:
        raise ConfigSyntaxError(f"Invalid value {value}, {err_msg}")


def _analyse_non_neg_number(value):
    number = _analyse_atomic_value(
        lambda x: isinstance(x, (float, int)) and x >= 0.0,
        value,
        "must be a non-negative `float` or `int` number",
    )
    return float(number)


def _analyse_0_to_1_number(value):
    number = _analyse_atomic_value(
        lambda x: isinstance(x, (float, int)) and x >= 0.0 and x <= 1.0,
        value,
        "must be a `float` or `int` number from [0, 1]",
    )
    return float(number)


def _analyse_half_to_1_number(value):
    number = _analyse_atomic_value(
        lambda x: isinstance(x, (float, int)) and x >= 0.5 and x <= 1.0,
        value,
        "must be a `float` or `int` number from [0.5, 1]",
    )
    return float(number)


def _analyse_non_neg_int(value):
    return _analyse_atomic_value(
        lambda x: isinstance(x, int) and x >= 0, value, "must be a non-negative `int`"
    )


def _analyse_positive_int(value):
    return _analyse_atomic_value(
        lambda x: isinstance(x, int) and x > 0, value, "must be a positive `int`"
    )


def _analyse_number(value):
    number = _analyse_atomic_value(
        lambda x: isinstance(x, (float, int)),
        value,
        "must be a `float` or `int` number",
    )
    return float(number)


def vectorized_append(lists, elems):
    for lst, elem in zip(lists, elems):
        lst.append(elem)

