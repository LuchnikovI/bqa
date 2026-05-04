from io import TextIOBase
import sys
import os
import logging
from pathlib import Path
from functools import singledispatch
from json import JSONDecodeError, load, dump

CWD = os.getcwd()
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]
DEFAULT_LOG_LEVEL = "INFO"


class CliError(Exception):
    pass


def resolve_path_to_json(path):
    try:
        p = Path(path).resolve()
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


def raise_unknown_arg_error(arg):
    raise CliError(f"Invalid command line argument {arg}, run `{sys.argv[0]} --help` to get the documentation")


def make_default_context():
    return {
        "input" : sys.stdin,
        "output" : sys.stdout,
        "log-level" : DEFAULT_LOG_LEVEL,
    }


@singledispatch
def read_json(src):
    raise TypeError(f"Unsupported destination: {type(src)}")


@read_json.register
def _(src: TextIOBase):
    try:
        return load(src)
    except (JSONDecodeError, UnicodeError, OSError) as e:
        raise CliError("Error while parsing json data") from e


@read_json.register
def _(src: Path):
    try:
        with src.open("r") as f:
            return load(f)
    except (JSONDecodeError, PermissionError, IsADirectoryError, UnicodeError, OSError) as e:
        raise CliError("Error while parsing json data") from e


@singledispatch
def dump_json(dst, data) -> None:
    raise TypeError(f"Unsupported destination: {type(dst)}")


@dump_json.register
def _(dst: TextIOBase, data):
    try:
        dump(data, dst)
    except (TypeError, OSError) as e:
        raise CliError("Error while writing result") from e


@dump_json.register
def _(dst: Path, data):
    try:
        with dst.open("w") as f:
            dump(data, f)
    except (TypeError, PermissionError, IsADirectoryError, OSError) as e:
        raise CliError("Error while writing result") from e


def format_error(e):
    msgs = []
    while e:
        msgs.append(str(e))
        e = e.__cause__
    return "\ncaused by: ".join(msgs)


class json_input_output_cli:
    def __init__(self, func):
        self.func = func
        self.name = func.__name__
        self.description = func.__doc__


    def print_help(self):
        print(f"""{self.description}
    
usage: poetry run {self.name[1:]} [options]
            
options:
 -i | --input  [PATH]         relative to `{CWD}` (current working directory) path to `*.json` config,
                              stdin is used if not provided
 -o | --output [PATH]         relative to `{CWD}` (current working directory) path to `*.json` where results are saved,
                              stdout is used if not provided
 -l | --log-level [LOG_LEVEL] logging level, the argument may take value from {", ".join(LOG_LEVELS)},
                              {DEFAULT_LOG_LEVEL} is used if not provided
 -h | --help:                 show this message and exit""")


    def dispatch(self, arg, args_iter, context):
        if arg in {"-h", "--help"}:
            self.print_help()
            sys.exit(0)
        elif arg in {"-i", "--input"}:
            input_path = next(args_iter, None)
            if input_path is None:
                raise CliError(f"Input file is not provided after {arg} key")
            context["input"] = resolve_path_to_json(input_path)
        elif arg in {"-o", "--output"}:
            output_path = next(args_iter, None)
            if output_path is None:
                raise CliError(f"Output file is not provided after {arg} key")
            context["output"] = resolve_path_to_json(output_path)
        elif arg in {"-l", "--log-level"}:
            log_level = next(args_iter, None)
            if log_level is None:
                raise CliError(f"Logging level is not provided after {arg} key, must be one of {', '.join(LOG_LEVELS)}")
            context["log-level"] = resolve_log_level(log_level)
        else:
            raise_unknown_arg_error(arg)


    def parse_args(self, argv):
        _, *args = argv
        args_iter = iter(args)
        context = make_default_context()
        while (arg := next(args_iter, None)) is not None:
            self.dispatch(arg, args_iter, context)
        return context


    def __call__(self):
        try:
            context = self.parse_args(sys.argv)
            set_log_level(context["log-level"])
            config = read_json(context["input"])
            result = self.func(config)
            dump_json(context["output"], result)
        except Exception as e:
            print(format_error(e), file=sys.stderr)
            sys.exit(1)        

