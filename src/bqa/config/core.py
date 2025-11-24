import logging
from bqa.config.config_canonicalization import Context, _canonicalize_config
from bqa.config.config_syntax import _analyse_config

log = logging.getLogger(__name__)

def config_to_context(config) -> Context:
    context = _canonicalize_config(_analyse_config(config))
    log.info("Context is built")
    return context
    
