from bqa.config.schedule_syntax import (ACTIONS_KEY, FINAL_MIXING_KEY, INITIAL_MIXING_KEY,
                                        SIMPLE_ACTION_TYPES, STEPS_NUMBER_KEY, TOTAL_TIME_KEY,
                                        TYPE_KEY, WEIGHT_KEY, COMPLEX_ACTION_TYPES)


def _interpolate_linearly(start, end, steps_number):
    delta = (end - start) / steps_number
    for n in range(steps_number):
        yield start + n * delta


def _canonicalize_complex_action(total_time, action):
    action_type = action[TYPE_KEY]
    final_mixing = action[FINAL_MIXING_KEY]
    initial_mixing = action[INITIAL_MIXING_KEY]
    steps_number = action[STEPS_NUMBER_KEY]
    time_step = total_time * action[WEIGHT_KEY] / steps_number
    for p in _interpolate_linearly(initial_mixing, final_mixing, steps_number):
        yield {"type" : action_type, "xtime": p * time_step, "ztime": (1.0 - p) * time_step}


def _canonicalize_schedule(schedule):
    total_time = schedule[TOTAL_TIME_KEY]
    actions = schedule[ACTIONS_KEY]
    for action in actions:
        action_type = action[TYPE_KEY]
        if action_type in COMPLEX_ACTION_TYPES:
            yield from _canonicalize_complex_action(total_time, action)
        elif action_type in SIMPLE_ACTION_TYPES:
            yield action_type
        else:
            assert False

