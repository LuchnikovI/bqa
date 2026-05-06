import logging
from statistics import mean, stdev
from bqa.config.compile_config import compile_config
from bqa.config.desugar_config import desugar_config, get_iter, get_ids_iter
from bqa.config.sparsify_config import get_nodes_number, sparsify_config
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


def get_values_iter(container):
    return map(lambda x: x[1], get_iter(container))


def get_metrics(config):
    config = full_preprocess(config)
    nodes_number = get_nodes_number(config[NODES_KEY], config[EDGES_KEY])
    edges_number = len(config[EDGES_KEY])
    edges = config[EDGES_KEY]
    max_cpl = max(get_values_iter(edges))
    min_cpl = min(get_values_iter(edges))
    stdev_cpl = stdev(get_values_iter(edges))
    mean_cpl = mean(get_values_iter(edges))
    mean_abs_cpl = mean(map(abs, get_values_iter(edges)))
    nodes = config[NODES_KEY]
    max_field = max(get_values_iter(nodes))
    min_field = min(get_values_iter(nodes))
    stdev_field = stdev(get_values_iter(nodes))
    mean_field = mean(get_values_iter(nodes))
    mean_abs_field = mean(map(abs, get_values_iter(nodes)))
    degrees = [0 for _ in range(nodes_number)]
    for li, ri in get_ids_iter(edges):
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
