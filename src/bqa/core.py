import os
from bqa.config.schedule_syntax import GET_BLOCH_VECTORS, MEASURE

os.environ["CUPY_ACCELERATORS"] = "cutensor"

import logging
from numpy.typing import NDArray
from bqa.config.core import config_to_context
from bqa.state import get_density_matrices, _initialize_state, measure, run_layer
from bqa.utils import convert_density_matrix_to_bloch_vector

log = logging.getLogger(__name__)

BLOCH_VECTORS_KEY = "bloch_vectors"

MEASUREMENT_OUTCOMES_KEY = "measurement_outcomes"

ALL_RESULT_KEYS = {BLOCH_VECTORS_KEY, MEASUREMENT_OUTCOMES_KEY}

def run_qa(config) -> list:
    context = config_to_context(config)
    instructions_number = len(context.instructions)
    state = _initialize_state(context)

    def execute_instruction(
            instruction_number: int,
            instruction: str | dict,
    ) -> None | list | NDArray:
        log.info(f"Instruction number {instruction_number} / {instructions_number} started")
        if isinstance(instruction, dict):
            return run_layer(context, instruction["xtime"], instruction["ztime"], state)
        elif instruction == MEASURE:
            return [MEASUREMENT_OUTCOMES_KEY, measure(context, state)]
        elif instruction == GET_BLOCH_VECTORS:
            return [
                BLOCH_VECTORS_KEY,
                list(map(convert_density_matrix_to_bloch_vector, get_density_matrices(context, state))),
            ]
        else:
            raise ValueError(f"Unknown instruction {instruction}")
    instr_exec_iter = (execute_instruction(instr_num, instr) for instr_num, instr in enumerate(context.instructions))
    return list(filter(lambda x: x is not None, instr_exec_iter))

