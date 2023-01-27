import logging

from .climate import _try_register_climate_modules
from .utils.registry import registry


def make_module(id_module, **kwargs):
    logging.info("initializing module {}".format(id_module))
    _module = registry.get_lightningmodule(id_module)
    assert _module is not None, "Could not find module with name {}".format(id_module)
    return _module(**kwargs)


_try_register_climate_modules()
