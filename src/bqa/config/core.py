import logging
from statistics import mean, stdev
from bqa.config.compile_config import compile_config
from bqa.config.desugar_config import desugar_config
from bqa.config.metrics import get_edges_number, get_nodes_number
from bqa.config.sparsify_config import sparsify_config
from bqa.config.validate_config import EDGES_KEY, NODES_KEY, validate_config

log = logging.getLogger(__name__)


def config_to_context(config):
    context =  config | validate_config | desugar_config | sparsify_config | compile_config
    log.info("Context is built")
    return context


def canonicalize(config):
    return config | validate_config | desugar_config

def full_preprocess(config):
    return config | validate_config | desugar_config | sparsify_config

def get_metrics(config):
    config = full_preprocess(config)
    nodes_number = get_nodes_number(config)
    edges_number = get_edges_number(config)
    edges = config[EDGES_KEY]
    max_cpl = max(edges.values())
    min_cpl = min(edges.values())
    stdev_cpl = stdev(edges.values())
    mean_cpl = mean(edges.values())
    mean_abs_cpl = mean(map(abs, edges.values()))
    nodes = config[NODES_KEY]
    max_field = max(nodes.values())
    min_field = min(nodes.values())
    stdev_field = stdev(nodes.values())
    mean_field = mean(nodes.values())
    mean_abs_field = mean(map(abs, nodes.values()))
    degrees = [0 for _ in range(nodes_number)]
    for li, ri in edges.keys():
        degrees[li] += 1
        degrees[ri] += 1
    max_degree = max(degrees)
    min_degree = min(degrees)
    stdev_degree = stdev(degrees)
    mean_degree = mean(degrees)
    return {
        "nodes_number" : nodes_number,
        "edges_number" : edges_number,
        "max_coupling" : max_cpl,
        "min_coupling" : min_cpl,
        "stddev_coupling" : stdev_cpl,
        "mean_coupling" : mean_cpl,
        "mean_abs_coupling" : mean_abs_cpl,
        "max_field" : max_field,
        "min_field" : min_field,
        "stddev_field" : stdev_field,
        "mean_field" : mean_field,
        "mean_abs_field" : mean_abs_field,
        "max_degree" : max_degree,
        "min_degree" : min_degree,
        "stddev_degree" : stdev_degree,
        "mean_degree" : mean_degree,
    }
