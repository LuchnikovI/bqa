import datetime
import json
from statistics import mean, stdev
from pathlib import Path
from mqlib_wrap import run_heuristics, get_energy_function
from bqa import run_qa

script_dir = Path(__file__).resolve().parent


def project_bloch_vectors(bloch_vectors):
    return list(1 if bv[2] > 0 else -1 for bv in bloch_vectors)


def extract_argmin_from_last_record(bptn_result):
    ty, res = bptn_result[-1]
    if ty == "bloch_vectors":
        return project_bloch_vectors(res)
    elif ty == "measurement_outcomes":
        return res
    else:
        raise ValueError(f"Cannot extract argmin from a result of type {ty}")


def get_some_graph_statistics(edges, nodes):
    nodes_number = max(max(*nodes.keys()), max(*(max(lhs, rhs) for lhs, rhs in edges.keys()))) + 1
    edges_number = len(edges)
    graph = [list() for _ in range(nodes_number)]
    for lhs, rhs in edges:
        graph[lhs].append(rhs)
        graph[rhs].append(lhs)
    degrees = list(len(nb) for nb in graph)
    mean_degree = mean(degrees)
    std_degree = stdev(degrees)
    mean_field = mean(nodes.values())
    std_field = stdev(nodes.values())
    mean_coupling = mean(edges.values())
    std_coupling = stdev(edges.values())
    return {
        "nodes_number" : nodes_number,
        "edges_number" : edges_number,
        "mean_degree" : mean_degree,
        "std_degree" : std_degree,
        "mean_field" : mean_field,
        "std_field" : std_field,
        "mean_coupling" : mean_coupling,
        "std_coupling" : std_coupling,
    }

def replace_nodes_and_edges_with_statistics(config):
    nodes = config["nodes"]
    edges = config["edges"]
    truncated_config = {k : v for k, v in config.items() if k not in {"nodes", "edges", "default_field"}}
    return truncated_config | get_some_graph_statistics(edges, nodes)


def run_benchmarks(config):
    output_dir = script_dir / config["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    bptn_result = run_qa(config)
    mqlib_results = run_heuristics(config)
    energy_function = get_energy_function(config)
    bptn_configuration = extract_argmin_from_last_record(bptn_result)
    bptn_energy = energy_function(bptn_configuration)
    all_results = [(name, result) for name, result in mqlib_results.items()]
    all_results.sort(key = lambda pair: pair[1]["energy"])
    config_and_results = {
        "config" : replace_nodes_and_edges_with_statistics(config),
        "bptn_result" : {"energy" : bptn_energy, "configuration" : bptn_configuration},
        "results" : all_results,
    }
    with open(output_dir / f"{timestamp}_result.json", "w") as file:
        json.dump(config_and_results, file)
