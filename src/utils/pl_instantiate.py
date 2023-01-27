from importlib import import_module
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LightningLoggerBase


def instantiate_class(init: Dict[str, Any]) -> Any:
    """Instantiates a class with the given args and init.
    Args:
        todo
    Returns:
        The instantiated class object.
    """
    kwargs = {k: init[k] for k in set(list(init.keys())) - {"_target_"}}

    class_module, class_name = init["_target_"].rsplit(".", 1)
    module = import_module(class_module, package=class_name)
    args_class = getattr(module, class_name)
    return args_class(**kwargs)


def instantiate_callbacks(callbacks_cfg: dict) -> List[Callback]:
    """Instantiates callbacks from config."""
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        return callbacks

    if not isinstance(callbacks_cfg, dict):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, dict) and "_target_" in cb_conf:
            callbacks.append(instantiate_class(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: dict) -> List[LightningLoggerBase]:
    """Instantiates loggers from config."""
    logger: List[LightningLoggerBase] = []

    if not logger_cfg:
        return logger

    if not isinstance(logger_cfg, dict):
        raise TypeError("Logger config must be a Dict!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, dict) and "_target_" in lg_conf:
            logger.append(instantiate_class(lg_conf))

    return logger


def instantiate_trainer(
    trainer_cfg: dict, callbacks_cfg: dict, logger_cfg: dict, seed: Optional[int] = None
):
    if seed is not None:
        pl.seed_everything(seed, workers=True)

    callbacks: List[Callback] = instantiate_callbacks(callbacks_cfg)
    logger: List[LightningLoggerBase] = instantiate_loggers(logger_cfg)
    trainer: Trainer = Trainer(**trainer_cfg, callbacks=callbacks, logger=logger)

    return trainer