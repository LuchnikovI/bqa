import logging

from bqa.config.compile_config import Context, compile_config
from bqa.config.desugar_config import desugar_config
from bqa.config.sparsify_config import sparsify_config
from bqa.config.validate_config import validate_config

log = logging.getLogger(__name__)


def config_to_context(config) -> Context:
    context =  config | validate_config | desugar_config | sparsify_config | compile_config
    log.info("Context is built")
    return context

