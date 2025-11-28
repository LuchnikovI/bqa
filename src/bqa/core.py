from functools import reduce
import logging
from typing import Any, Callable
from bqa.config.core import config_to_context
from bqa.config.schedule_canonicalization import Instruction
from bqa.state import State, _get_density_matrices, _initialize_state, measure, run_layer

log = logging.getLogger(__name__)

def run_qa(config) -> list:
    context = config_to_context(config)
    instructions_number = len(context.instructions)

    def execute_instruction(acc: dict, instruction_number_and_instruction: tuple[int, Instruction]) -> dict:
        instruction_number, instruction = instruction_number_and_instruction
        log.info(f"Instruction number {instruction_number} / {instructions_number} started")

        def apply_func_to_state(func: Callable[[State], Any], acc: dict) -> dict:
            output = func(acc["state"])
            if output is not None:
                acc["outputs"].append(output)
            return acc

        if isinstance(instruction, dict):
            return apply_func_to_state(lambda state: run_layer(context, instruction["xtime"], instruction["ztime"], state), acc)
        elif instruction == "measure":
            return apply_func_to_state(lambda state: measure(context, state), acc)
        elif instruction == "get_density_matrices":
            return apply_func_to_state(lambda state: _get_density_matrices(context, state), acc)
        else:
            raise ValueError(f"Unknown instruction {instruction}")

    def extract_outputs(acc: dict) -> list:
        return acc["outputs"]

    acc = {"state" : _initialize_state(context), "outputs" : []}

    return extract_outputs(reduce(execute_instruction, enumerate(context.instructions), acc))
