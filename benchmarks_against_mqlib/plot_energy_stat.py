#!/usr/bin/env python3
import sys
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent

def gen_file_paths(dir_name):
    full_dir_path = SCRIPT_DIR / dir_name
    for full_name in map(lambda name: full_dir_path / name, os.listdir(full_dir_path)):
        yield full_name


def read_jsons(json_files):
    for json_file in json_files:
        with open(json_file, "r") as file:
            yield json.load(file)


def main():
    _, dir_name = sys.argv
    name_to_sum_normalized_energy = {}
    for n, stat in enumerate(read_jsons(gen_file_paths(dir_name))):
        name_to_energy = {name : result["energy"] for name, result in stat["results"]}
        name_to_energy["QA"] = stat["bptn_result"]["energy"]
        max_e = max(name_to_energy.values())
        min_e = min(name_to_energy.values())
        delta = max_e - min_e
        for name, e in name_to_energy.items():
            normalized_e = 1. if delta == 0 else (max_e - e) / delta
            if name not in name_to_sum_normalized_energy:
                name_to_sum_normalized_energy[name] = normalized_e
            else:
                name_to_sum_normalized_energy[name] *= n / (n + 1)
                name_to_sum_normalized_energy[name] += normalized_e / (n + 1)
    names, values = zip(*sorted(name_to_sum_normalized_energy.items(), key=lambda x: x[1]))
    fig, ax = plt.subplots()
    ax.plot(names, values, "-b")
    ax.set_ylabel("relative_energy")
    ax.tick_params(axis="x", labelrotation=90)
    for label in ax.get_xticklabels():
        if label.get_text() == "QA":
            label.set_fontweight("bold")
    fig.tight_layout()
    plt.savefig(SCRIPT_DIR / f"{dir_name}.pdf")


if __name__ == "__main__":
    main()

