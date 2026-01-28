#!/usr/bin/env python3

import sys
import logging
from json import dumps
from bqa.benchmarking import generate_qubo_on_2d_grid, generate_qubo_on_random_regular_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

NEW_LINE = "\n"

INDEN = "\t"

CLI_SPEC = {
    "--backend" : {
        "default" : "numpy",
        "help" : "computational backend (either numpy or cupy)",
        "parser" : str,
    },
    "--seed" : {
        "default" : 42,
        "help" : "random seed used to sample measurements",
        "parser" : int,
    },
    "--damping" : {
        "default" : 0.0,
        "help" : "exponential moving average in BP algorithm during measurements sampling stage",
        "parser" : float,
    },
    "--max_bond_dim" : {
        "default" : 4,
        "help" : "maximal bond dimension allowed during simulation",
        "parser" : int,
    },
    "--bp_eps" : {
        "default" : 1e-10,
        "help" : "convergence threshold of the BP algorithm",
        "parser" : float,
    },
    "--pinv_eps" : {
        "default" : 1e-10,
        "help" : "close to zero singular value truncation threshold",
        "parser" : float,
    },
    "generators" : {
        "random_regular" : {
            "exec" : lambda nodes_number, degree: generate_qubo_on_random_regular_graph(nodes_number, degree = degree),
            "help" : "generates a random 3-regular graph with randomly selected local fields and interaction constants",
            "args" : {
                "--nodes_number" : {
                    "help" : "number of nodes in a graph",
                    "default" : 1000,
                    "parser" : int,
                },
                "--degree" : {
                    "help" : "degree of a generated graph",
                    "default" : 3,
                    "parser" : int,
                },
            },
        },
        "2d_grid" : {
            "exec" : generate_qubo_on_2d_grid,
            "help" : "generates a two-dimensional square lattice with randomly selected local fields and interaction constants",
            "args" : {
                "--m" : {
                    "help" : "vertical size of a grid (nodes number)",
                    "default" : 10,
                    "parser" : int,
                },
                "--n" : {
                    "help" : "horizontal size of a grid (nodes number)",
                    "default" : 10,
                    "parser" : int,
                },
            },
        },
    },
}


def gen_help_header(name):
    return f"""This is a CLI tool to generate *.json QA specification to benchmark a simulator performance.
usage: {name} --help | *parameters generator *generator-args"""


def get_arg_help(spec, name, indent = 1):
    return f"{indent * INDEN}{name}: {spec['help']}, default value: {spec['default']}{NEW_LINE}"


def gen_params_help():
    yield "Parameters (usage: parameter_name parameter_value):\n"
    for name, spec in CLI_SPEC.items():
        if name != "generators":
            yield get_arg_help(spec, name = name)


def gen_generators_help():
    yield "Generators (usage: generator_name arg1-name arg1 arg2-name arg2 ...):\n"
    generators = CLI_SPEC["generators"]
    for name, spec in generators.items():
        yield f"{INDEN}{name}: {spec['help']}, args:{NEW_LINE}"
        for name, spec in spec["args"].items():
            yield get_arg_help(spec, name, indent = 2)


def gen_help(name):
    yield gen_help_header(name)
    yield "\n\n"
    yield from gen_params_help()
    yield "\n"
    yield from gen_generators_help()


def get_help(name):
    return "".join(gen_help(name))


def get_parameters_list():
    return list(p for p in CLI_SPEC.keys() if p != "generators")


def get_generators_list():
    return list(CLI_SPEC["generators"].keys())


def get_default_config():
    default_config = {}
    for name, subspec in CLI_SPEC.items():
        if name != "generators":
            default_config[name[2:]] = subspec["default"]
    return default_config


def get_default_args(spec):
    default_args = {}
    for name, subspec in spec.items():
        val = subspec["default"]
        default_args[name[2:]] = val
    return default_args


def make_coursor(lst):
    return [0, lst]


def get_current(coursor):
    ptr, lst = coursor
    return lst[ptr]


def is_not_empty(coursor):
    ptr, lst = coursor
    return ptr < len(lst)


def incr(coursor):
    coursor[0] += 1


def update_default(coursor, default_config, spec):
    name = get_current(coursor)
    assert name is not None
    if name not in spec:
        print(f"Unknown argument {name}, must be one from the list {list(spec.keys())}")
        exit(1)
    parse = spec[name]["parser"]
    incr(coursor)
    val = get_current(coursor)
    if val is None:
        print(f"Value of {name} is not provided")
        exit(1)
    default_config[name[2:]] = parse(val)
    incr(coursor)


def insert_graph(generator, config, coursor):
    func = generator["exec"]
    args = generator["args"]
    default_args = get_default_args(args)
    while is_not_empty(coursor):
        update_default(coursor, default_args, args)
    nodes, edges = func(*default_args.values())
    config["nodes"] = list([k, v] for k, v in nodes.items())
    config["edges"] = list([[lk, rk], v] for (lk, rk), v in edges.items())


def parse_cli_args_to_config(coursor):
    default_config = get_default_config()
    while is_not_empty(coursor):
        param = get_current(coursor)
        if param in CLI_SPEC["generators"]:
            generator = CLI_SPEC["generators"][param]
            incr(coursor)
            insert_graph(generator, default_config, coursor)
            return default_config
        elif param in CLI_SPEC:
            update_default(coursor, default_config, CLI_SPEC)
        else:
                print(f"Unknown parameter / generator {param}, must be either from the parameters list {get_parameters_list()} or from the generators list {get_generators_list()}")
                exit(1)
    print("Generator is not provided")
    exit(1)

def main():
    name, *args = sys.argv
    coursor = make_coursor(args)
    if not is_not_empty(coursor):
        print("Empty CLI arguments, see the help message:\n\n")
        print(get_help(name))
        exit(1)
    elem = get_current(coursor)
    if elem == "--help":
        print(get_help(name))
        exit(0)
    else:
        print(dumps(parse_cli_args_to_config(coursor), indent=2))

if __name__ == "__main__":
    main()

