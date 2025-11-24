from math import isclose
from typing import Optional
from .utils import (
    ConfigSyntaxError,
    _analyse_0_to_1_number,
    _analyse_non_neg_number,
    _analyse_positive_int,
    _filtermap,
    _ok_or_default_and_warn,
    _is_sequential,
    _unwrap_or,
)

# types

LinearScaling = dict[str, float | int]

Action = LinearScaling | str

Actions = list[Action]

Schedule = dict[str, Actions | float]

# defaults

DEFAULT_TOTAL_TIME = 10.0

DEFAULT_STEPS_NUMBER = 100

DEFAULT_TIME = 1.0

DEFAULT_STARTING_MIXING = 1.0

DEFAULT_FINAL_MIXING = 0.0

DEFAULT_LINEAR_SCALING = {
    "time": DEFAULT_TIME,
    "steps_number": DEFAULT_STEPS_NUMBER,
    "final_mixing": DEFAULT_FINAL_MIXING,
}

DEFAULT_ACTIONS = [
    DEFAULT_LINEAR_SCALING,
    "get_observables",
]

DEFAULT_SCHEDULE = {
    "total_time": DEFAULT_TOTAL_TIME,
    "starting_mixing": DEFAULT_STARTING_MIXING,
    "actions": DEFAULT_ACTIONS,
}

# syntax analysis


def _analyse_linear_scaling(linear_scaling) -> LinearScaling:
    time = _unwrap_or(linear_scaling.get("time"), "`time` field is missing")
    steps_number = _ok_or_default_and_warn(
        linear_scaling.get("steps_number"),
        DEFAULT_STEPS_NUMBER,
        f"`steps number` is missing in linear scaling, set to {DEFAULT_STEPS_NUMBER}",
    )
    final_mixing = _unwrap_or(
        linear_scaling.get("final_mixing"), "`final_mixing` field is missing"
    )
    return {
        "time": _analyse_0_to_1_number(time),
        "steps_number": _analyse_positive_int(steps_number),
        "final_mixing": _analyse_0_to_1_number(final_mixing),
    }


def _analyse_action(action) -> Action:
    if isinstance(action, dict):
        try:
            return _analyse_linear_scaling(action)
        except ConfigSyntaxError as e:
            raise ConfigSyntaxError(f"Invalid linear scaling {action}") from e
    elif action == "get_observables" or action == "measure":
        return action
    else:
        raise ConfigSyntaxError(f"Unknown action {action}")


def _extract_time(action: Action) -> Optional[float]:
    if isinstance(action, dict):
        return action["time"]


def _check_times_sum_to_one(actions: Actions) -> None:
    if not isclose(sum(_filtermap(actions, _extract_time)), 1.0):
        raise ConfigSyntaxError(f"`time` fields must sum into 1. in actions {actions}")


def _analyse_actions(actions) -> Actions:
    if _is_sequential(actions):
        try:
            actions = list(map(_analyse_action, actions))
            _check_times_sum_to_one(actions)
            return actions
        except ConfigSyntaxError as e:
            raise ConfigSyntaxError(f"Invalid actions {actions}") from e
    else:
        raise ConfigSyntaxError(f"Invalid actions {actions}")


def _analyse_schedule(schedule) -> Schedule:
    if isinstance(schedule, dict):
        try:
            total_time = _analyse_non_neg_number(
                _ok_or_default_and_warn(
                    schedule.get("total_time"),
                    DEFAULT_TOTAL_TIME,
                    f"`total_time` field is missing in the schedule, set to {DEFAULT_TOTAL_TIME}",
                )
            )
            starting_mixing = _analyse_0_to_1_number(
                _ok_or_default_and_warn(
                    schedule.get("starting_mixing"),
                    DEFAULT_STARTING_MIXING,
                    f"`starting_mixing` field is missing in the schedule, \
                    set to {DEFAULT_STARTING_MIXING}",
                )
            )
            actions = _analyse_actions(
                _ok_or_default_and_warn(
                    schedule.get("actions"),
                    DEFAULT_ACTIONS,
                    f"`actions` field is missing in the schedule, set to {DEFAULT_ACTIONS}",
                )
            )
            return {
                "total_time": total_time,
                "starting_mixing": starting_mixing,
                "actions": actions,
            }
        except ConfigSyntaxError as e:
            raise ConfigSyntaxError(f"Invalid schedule {schedule}") from e
    else:
        raise ConfigSyntaxError(f"Invalid schedule {schedule}")
