from math import isclose
from typing import Optional
from .utils import (ConfigSyntaxError, analyse_0_to_1_float, analyse_non_neg_float,
                    analyse_positive_int, filtermap, get_default_and_warn, is_sequential,
                    unwrap_or)

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

DEFAULT_LINEAR_SCALING = {"time" : DEFAULT_TIME,
                          "steps_number" : DEFAULT_STEPS_NUMBER,
                          "final_mixing" : DEFAULT_FINAL_MIXING}

DEFAULT_ACTIONS = [
    DEFAULT_LINEAR_SCALING,
    "get_observables",
]

DEFAULT_SCHEDULE = {"total_time" : DEFAULT_TOTAL_TIME,
                    "starting_mixing" : DEFAULT_STARTING_MIXING,
                    "actions" : DEFAULT_ACTIONS}

# syntax analysis

def analyse_linear_scaling(linear_scaling) -> LinearScaling:
    time = unwrap_or(linear_scaling.get("time"), "`time` field is missing")
    steps_number = linear_scaling.get("steps_number") \
        or get_default_and_warn(DEFAULT_STEPS_NUMBER,
        f"`steps number` is missing in linear scaling, set to {DEFAULT_STEPS_NUMBER}")
    final_mixing = unwrap_or(linear_scaling.get("final_mixing"), "`final_mixing` field is missing")
    return {"time" : analyse_0_to_1_float(time),
            "steps_number" : analyse_positive_int(steps_number),
            "final_mixing" : analyse_0_to_1_float(final_mixing)}

def analyse_action(action) -> Action:
    if isinstance(action, dict):
        try:
            return analyse_linear_scaling(action)
        except ConfigSyntaxError as e:
            raise ConfigSyntaxError(f"Invalid linear scaling {action}") from e
    elif action == "get_observables" or action == "measure":
        return action
    else:
        raise ConfigSyntaxError(f"Unknown action {action}")

def extract_time(action: Action) -> Optional[float]:
    if isinstance(action, dict):
        return action["time"]

def check_times_sum_to_one(actions: Actions) -> None:
    if not isclose(sum(filtermap(actions, extract_time)), 1.):
        raise ConfigSyntaxError(f"`time` fields must sum into 1. in actions {actions}")

def analyse_actions(actions) -> Actions:
    if is_sequential(actions):
        try:
            actions = list(map(analyse_action, actions))
            check_times_sum_to_one(actions)
            return actions
        except ConfigSyntaxError as e:
            raise ConfigSyntaxError(f"Invalid actions {actions}") from e
    else:
        raise ConfigSyntaxError(f"Invalid actions {actions}")

def analyse_schedule(schedule) -> Schedule:
    if isinstance(schedule, dict):
        try:
            total_time = analyse_non_neg_float(
                schedule.get("total_time")
                or get_default_and_warn(
                    DEFAULT_TOTAL_TIME,
                    f"`total_time` field is missing in the schedule, set to {DEFAULT_TOTAL_TIME}",
                )
            )
            starting_mixing = analyse_0_to_1_float(
                schedule.get("starting_mixing")
                or get_default_and_warn(
                    DEFAULT_STARTING_MIXING,
                    f"`starting_mixing` field is missing in the schedule, set to {DEFAULT_STARTING_MIXING}"
                )
            )
            actions = analyse_actions(
                schedule.get("actions")
                or get_default_and_warn(
                    DEFAULT_ACTIONS,
                    f"`actions` field is missing in the schedule, set to {DEFAULT_ACTIONS}",
                )
            )
            return {"total_time" : total_time,
                    "starting_mixing" : starting_mixing,
                    "actions" : actions}
        except ConfigSyntaxError as e:
            raise ConfigSyntaxError(f"Invalid schedule {schedule}") from e
    else:
        raise ConfigSyntaxError(f"Invalid schedule {schedule}")

