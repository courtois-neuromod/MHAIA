# Re-export config functions from CL.config for compatibility
from CL.config import update_wrapper_config, get_arg_parser

__all__ = ['update_wrapper_config', 'get_arg_parser']
