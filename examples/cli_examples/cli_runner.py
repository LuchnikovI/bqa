#!/usr/bin/env python3

import sys
import logging
from pathlib import Path
from json import load, dump
from bqa import run_qa

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

SCRIPT_DIR = Path(__file__).resolve().parent


def print_help(name):
    print(f"""usage: {name} INPUT_JSON_PATH [OUTPUT_JSON_PATH] | --help | -h

    INPUT_JSON_PATH:  relative to `{SCRIPT_DIR}` path to the `*.json` config
    OUTPUT_JSON_PATH: relative to `{SCRIPT_DIR}` directory path where result `result.json` is saved,
                      default to `./`
    --help | -h:      show this message and exit""")

    
def print_error(name, args):
    print(f"Invalid command line arguments {args}, run `{name} --help` to get documentation")


def run(src_path, dst_path = str(SCRIPT_DIR)):
    src_full_path = (SCRIPT_DIR / src_path).resolve()
    dst_dir_full_path = (SCRIPT_DIR / dst_path).resolve()
    dst_dir_full_path.mkdir(parents=True, exist_ok=True)
    dst_full_path = (dst_dir_full_path / "result.json").resolve()
    with open(src_full_path, "r") as conf_file:
        conf = load(conf_file)
    result = run_qa(conf)
    with open(dst_full_path, "w") as res_file:
        dump(result, res_file, indent=2)

def main():
    name, *args = sys.argv
    if len(args) == 1:
        if (args[0] == "-h" or args[0] == "--help"):
            print_help(name)
        else:
            src_path = args[0]
            run(src_path)
    elif len(args) > 1:
        src_path, dst_path, *_ = args
        run(src_path, dst_path)
    else:
        print_error(name, args)
        exit(1)

if __name__ == "__main__":
    main()
