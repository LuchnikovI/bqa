from math import isclose
import logging
from .utils import (
    ConfigSyntaxError,
    _analyse_0_to_1_number,
    _analyse_non_neg_number,
    _analyse_positive_int,
    _is_sequential,
    _get_or_default_and_warn,
    _get_or_raise,
)

log = logging.getLogger(__name__)

# keywords

FINAL_MIXING_KEY = "final_mixing"

INITIAL_MIXING_KEY = "initial_mixing"

STARTING_MIXING_KEY = "starting_mixing"

TYPE_KEY = "type"

TOTAL_TIME_KEY = "total_time"

ACTIONS_KEY = "actions"

WEIGHT_KEY = "weight"

STEPS_NUMBER_KEY = "steps_number"

# action types

GET_DENSITY_MATRICES = "get_density_matrices"

MEASURE = "measure"

REAL_TIME_EV_TYPE = "real_time_evolution"

IMAG_TIME_EV_TYPE = "imag_time_evolution"

SIMPLE_ACTION_TYPES = {MEASURE, GET_DENSITY_MATRICES}

COMPLEX_ACTION_TYPES = {REAL_TIME_EV_TYPE, IMAG_TIME_EV_TYPE}

ACTION_TYPES = {REAL_TIME_EV_TYPE, IMAG_TIME_EV_TYPE, MEASURE, GET_DENSITY_MATRICES}

# defaults

DEFAULT_TOTAL_TIME = 10.0

DEFAULT_STEPS_NUMBER = 100

DEFAULT_WEIGHT = 1.0

DEFAULT_STARTING_MIXING = 1.0

DEFAULT_FINAL_MIXING = 0.0

DEFAULT_EVOLUTION_TYPE = "real_time_evolution"

DEFAULT_EVOLUTION = {
    TYPE_KEY : DEFAULT_EVOLUTION_TYPE,
    WEIGHT_KEY : DEFAULT_WEIGHT,
    STEPS_NUMBER_KEY : DEFAULT_STEPS_NUMBER,
    FINAL_MIXING_KEY : DEFAULT_FINAL_MIXING,
}

SIMPLE_ACTIONS = {MEASURE, GET_DENSITY_MATRICES}

DEFAULT_ACTIONS = [
    DEFAULT_EVOLUTION,
    GET_DENSITY_MATRICES,
]

DEFAULT_SCHEDULE = {
    "total_time": DEFAULT_TOTAL_TIME,
    "starting_mixing": DEFAULT_STARTING_MIXING,
    "actions": DEFAULT_ACTIONS,
}

# syntax analysis

def _check_weight_sum_to_one(actions):
    time_sum = sum(map(lambda x: x[WEIGHT_KEY], actions))
    if not isclose(time_sum, 1.0):
        raise ConfigSyntaxError(f"`{WEIGHT_KEY}` fields must sum into 1. in actions {actions}, but now it sums into {time_sum}")


def _desug_actions_seq(actions, starting_mixing):
    current_mixing = starting_mixing

    def desug_simple_action(action):
        return {
            TYPE_KEY : action,
            WEIGHT_KEY : 0,
            INITIAL_MIXING_KEY : current_mixing,
            FINAL_MIXING_KEY : current_mixing,
            STEPS_NUMBER_KEY : 1,
        }

    def desug_complex_action(action):
        nonlocal current_mixing
        copied_action = action.copy()
        if INITIAL_MIXING_KEY in copied_action:
            raise ConfigSyntaxError(f"`{INITIAL_MIXING_KEY}` must not be present in the action {copied_action}, it is infered automatically")
        copied_action[INITIAL_MIXING_KEY] = current_mixing
        if FINAL_MIXING_KEY not in copied_action:
            copied_action[FINAL_MIXING_KEY] = current_mixing
        else:
            current_mixing = copied_action[FINAL_MIXING_KEY]
        return copied_action

    for action in actions:
        if isinstance(action, dict):
            yield desug_complex_action(action)
        elif isinstance(action, str):
            yield desug_simple_action(action)
        else:
            raise TypeError(f"Invalid type {type(action)} of the action {action}")


def _analyse_actions_seq(actions):
    for action in actions:
        action_type = _get_or_default_and_warn(action, TYPE_KEY, REAL_TIME_EV_TYPE, "action")
        if action_type not in ACTION_TYPES:
            raise ConfigSyntaxError(f"Invalid action type {action_type} in action {action}")
        weight = _analyse_0_to_1_number(_get_or_raise(action, WEIGHT_KEY, "action"))
        steps_number = _analyse_positive_int(_get_or_default_and_warn(action, STEPS_NUMBER_KEY, DEFAULT_STEPS_NUMBER, "action"))
        initial_mixing = _analyse_0_to_1_number(action[INITIAL_MIXING_KEY])
        final_mixing = _analyse_0_to_1_number(action[FINAL_MIXING_KEY])
        yield {TYPE_KEY : action_type,
               WEIGHT_KEY : weight,
               STEPS_NUMBER_KEY : steps_number,
               INITIAL_MIXING_KEY : initial_mixing,
               FINAL_MIXING_KEY : final_mixing}


def _analyse_actions(actions, mixing):
    if _is_sequential(actions):
        try:
            analised_actions = list(_analyse_actions_seq(_desug_actions_seq(actions, mixing)))
            _check_weight_sum_to_one(analised_actions)
            return analised_actions
        except ConfigSyntaxError as e:
            raise ConfigSyntaxError(f"Invalid actions {actions}") from e
    else:
        raise ConfigSyntaxError(f"Actions must be either a list or a tuple, got {actions} of type {type(actions)}")


def _analyse_schedule(schedule):
    if isinstance(schedule, dict):
        try:
            total_time = _analyse_non_neg_number(_get_or_default_and_warn(schedule, TOTAL_TIME_KEY, DEFAULT_TOTAL_TIME, "schedule"))
            starting_mixing = _analyse_0_to_1_number(_get_or_default_and_warn(schedule, STARTING_MIXING_KEY, DEFAULT_STARTING_MIXING, "schedule"))
            actions = _analyse_actions(
                _get_or_default_and_warn(schedule, ACTIONS_KEY, DEFAULT_ACTIONS, "schedule"),
                starting_mixing,
            )
            return {
                TOTAL_TIME_KEY : total_time,
                ACTIONS_KEY : actions,
            }
        except ConfigSyntaxError as e:
            raise ConfigSyntaxError(f"Invalid schedule {schedule}") from e
    else:
        raise ConfigSyntaxError(f"Schedule must be a dict, got {schedule} of type {type(schedule)}")

