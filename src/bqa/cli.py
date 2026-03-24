#!/usr/bin/env python3

from io import TextIOBase
import sys
import os
import logging
from pathlib import Path
from functools import singledispatch
from json import JSONDecodeError, load, dump
from bqa import run_qa

CWD = os.getcwd()
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]
DEFAULT_LOG_LEVEL = "INFO"


def print_help():
    print(f"""usage: poetry run bqa_cli [options]

    options:
      -i | --input  [PATH]         relative to `{CWD}` (current working directory) path to `*.json` config,
                                   stdin is used if not provided
      -o | --output [PATH]         relative to `{CWD}` (current working directory) path to `*.json` where results are saved,
                                   stdout is used if not provided
      -l | --log-level [LOG_LEVEL] logging level, the argument may take value from {LOG_LEVELS},
                                   {DEFAULT_LOG_LEVEL} is used if not provided
      -h | --help:                 show this message and exit""")


class CliError(Exception):
    pass


def print_unknown_arg_error(arg):
    print(f"Invalid command line argument {arg}, run `{sys.argv[0]} --help` to get the documentation")


def resolve_path_to_json(path):
    try:
        p = (CWD / Path(path)).resolve()
    except (RuntimeError, OSError) as e:
        raise CliError(f"Error while checking {path} file") from e
    if p.suffix != ".json":
        raise CliError(f"{p} must have .json suffix")
    return p


def resolve_log_level(log_level):
    if log_level in LOG_LEVELS:
        return log_level
    else:
        raise CliError(f"{log_level} cannot be interpeted as a logging level, must be one of {', '.join(LOG_LEVELS)}")


def set_log_level(log_level):
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def dispatch(arg, args_iter, context):
    if arg in {"-h", "--help"}:
        print_help()
        exit(0)
    elif arg in {"-i", "--input"}:
        input_path = next(args_iter, None)
        if input_path is None:
            raise CliError(f"Input file is not provided after {arg} key")
        context["input"] = resolve_path_to_json(input_path)
        _parse_args(args_iter, context)
    elif arg in {"-o", "--output"}:
        output_path = next(args_iter, None)
        if output_path is None:
            raise CliError(f"Output file is not provided after {arg} key")
        context["output"] = resolve_path_to_json(output_path)
        _parse_args(args_iter, context)
    elif arg in {"-l", "--log-level"}:
        log_level = next(args_iter, None)
        if log_level is None:
            raise CliError(f"Logging level is not provided after {arg} key, must be one of {', '.join(LOG_LEVELS)}")
        context["log-level"] = resolve_log_level(log_level)
        _parse_args(args_iter, context)
    else:
        print_unknown_arg_error(arg)
        exit(1)


def _parse_args(args_iter, context):
    arg = next(args_iter, None)
    if arg is None:
        return
    else:
        dispatch(arg, args_iter, context)


def make_default_context():
    return {
        "input" : sys.stdin,
        "output" : sys.stdout,
        "log-level" : DEFAULT_LOG_LEVEL,
    }


def parse_args(argv):
    _, *args = argv
    args_iter = iter(args)
    context = make_default_context()
    _parse_args(args_iter, context)
    return context


@singledispatch
def read_json(src):
    raise TypeError(f"Unsupported destination: {type(src)}")


@read_json.register
def _(src: TextIOBase):
    try:
        return load(src)
    except (UnicodeError, OSError) as e:
        raise CliError("Error while parsing json data") from e


@read_json.register
def _(src: Path):
    try:
        print(src)
        with src.open("r") as f:
            return load(f)
    except (JSONDecodeError, PermissionError, IsADirectoryError, UnicodeError, OSError) as e:
        raise CliError("Error while parsing json data") from e


@singledispatch
def dump_json(dst, _) -> None:
    raise TypeError(f"Unsupported destination: {type(dst)}")


@dump_json.register
def _(dst: TextIOBase, data):
    try:
        dump(data, dst)
    except OSError as e:
        raise CliError("Error while writing result") from e


@dump_json.register
def _(dst: Path, data):
    try:
        with dst.open("w") as f:
            dump(data, f)
    except (PermissionError, IsADirectoryError, OSError) as e:
        raise CliError("Error while writing result") from e


def format_error(e):
    msgs = []
    while e:
        msgs.append(str(e))
        e = e.__cause__
    return "\ncaused by: ".join(msgs)


def cli():
    try:
        context = parse_args(sys.argv)
        set_log_level(context["log-level"])
        config = read_json(context["input"])
        result = run_qa(config)
        dump_json(context["output"], result)
    except Exception as e:
        print(format_error(e), file=sys.stderr)
        sys.exit(1)

