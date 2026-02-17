## What is it?
This directory contains artifacts for experimenting with a quantum annealing (QA) specification format built on top of the JSON format.

## Artifacts

1) `cli_runner.py` -- a small CLI app that takes a JSON based specification of QA, executes it, and saves the result. Run `./cli_runner.py --help` for the documentation;
2) `small_ibm_heavy_hex.json` -- an example JSON specification for QA that includes all available parameters. Most parameters are optional; if omitted, they are set to default values and corresponding warning messages are issued;
3) `json_generator.py` -- a small CLI app that generates a JSON specifications for QA with the most important parameters. Run `./json_generator.py --help` for the documentation. Usage example `./json_generator.py --max_bond_dim 5 random_regular --degree 3 --nodes_number 1000 > spec.json`, it generates a `spec.json` file containing a QA specification for a random 3-regular graph with 1000 nodes (qubits) and a maximum bond dimension of 5 during the simulation.
