from typing import Iterable
from bqa.config.schedule_syntax import Action, Actions, Schedule

Instruction = str | dict[str, float]


def _extract_mixings(starting_mixing: float, actions: Actions) -> Iterable[float]:
    mixing = starting_mixing
    yield mixing
    for action in actions:
        if isinstance(action, dict):
            mixing = action["final_mixing"]
        yield mixing


def _interpolate_linearly(
    start: float, end: float, steps_number: int
) -> Iterable[float]:
    delta = (end - start) / steps_number
    for n in range(steps_number):
        yield start + n * delta


def _canonicalize_action(
    total_time: float,
    starting_mixing: float,
    action: Action,
) -> Iterable[Instruction]:
    if isinstance(action, dict):
        final_mixing = action["final_mixing"]
        steps_number = action["steps_number"]
        assert isinstance(steps_number, int)
        time_step = total_time * action["time"] / steps_number
        yield from map(
            lambda p: {"xtime": p * time_step, "ytime": (1.0 - p) * time_step},
            _interpolate_linearly(starting_mixing, final_mixing, steps_number),
        )
    else:
        yield action


def _canonicalize_schedule(schedule: Schedule) -> Iterable[Instruction]:
    total_time = schedule["total_time"]
    assert isinstance(total_time, float)
    starting_mixing = schedule["starting_mixing"]
    assert isinstance(starting_mixing, float)
    actions = schedule["actions"]
    assert isinstance(actions, list)
    mixings = _extract_mixings(starting_mixing, actions)
    for action, mixing in zip(actions, mixings):
        yield from _canonicalize_action(total_time, mixing, action)
