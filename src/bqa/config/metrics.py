from bqa.config.validate_config import NODES_KEY, EDGES_KEY


def get_nodes_number(config):
    return 1 + max(
        max(config[NODES_KEY].keys(), default=-1),
        max((max(lhs, rhs) for lhs, rhs in config[EDGES_KEY].keys()), default = -1),
    )

def get_edges_number(config):
    return len(config[EDGES_KEY])
