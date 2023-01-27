from typing import List

from omegaconf import OmegaConf


def parse(args_list: List[str]):
    OmegaConf.register_new_resolver(
        "eval", eval
    )  # capble of evaluating simple expression in configs
    cfg = OmegaConf.from_cli(args_list)

    if "base" in cfg:
        basecfg = OmegaConf.load(cfg.base)
        del cfg.base
        cfg = OmegaConf.merge(basecfg, cfg)
        cfg = OmegaConf.to_container(cfg, resolve=True)
    else:
        raise SystemExit("Base configuration file not specified! Exiting.")

    return cfg