from importlib.metadata import version

__version__ = version("bqa")

from bqa.core import run_qa
from bqa.config.validate_config import validate_config
