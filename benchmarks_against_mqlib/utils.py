import datetime
import json
from pathlib import Path
from mqlib_wrap import run_heuristics, get_energy_function
from bqa import run_qa

script_dir = Path(__file__).resolve().parent

def run_benchmarks(config):
    output_dir = script_dir / config["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    bptn_result = run_qa(config)
    mqlib_results = run_heuristics(config)
    energy_function = get_energy_function(config)
    qa_energy = energy_function(bptn_result[0][1])
    all_results = [(name, result["energy"]) for name, result in mqlib_results.items()]
    all_results.append(("QA", qa_energy))
    all_results.sort(key = lambda pair: pair[1])
    truncated_config = {k : v for k, v in config.items() if k not in {"nodes", "edges", "default_field"}}
    config_and_results = {"config" : truncated_config, "results" : all_results}
    with open(output_dir / f"{timestamp}_result.json", "w") as file:
        json.dump(config_and_results, file, indent=4)
