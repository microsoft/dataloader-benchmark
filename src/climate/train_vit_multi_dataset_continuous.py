import os
import sys

import src
from src.utils import cli, pl_instantiate
from src.utils.registry import registry


def main():
    # skip the program name in sys.argv
    cfg = cli.parse(sys.argv[1:])
    trainer = pl_instantiate.instantiate_trainer(
        cfg["trainer"],
        cfg["callbacks"],
        None,  # no loggers for now
        cfg.get("seed_everything", None),
    )
    # model = registry.get_lightningmodule(cfg["model_name"])(**cfg["model"])
    model = registry.get_lightningmodule(cfg["model_name"])()  # empty model does not need parameters
    datamodule = registry.get_datamodule(cfg["data_name"])(**cfg["data"])
    datamodule.set_patch_size(cfg["model"]["net"]["init_args"]["patch_size"])
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()


# def main():
#     # Initialize Lightning with the model and data modules, and instruct it to parse the config yml
#     cli = LightningCLI(
#         model_class=ViTContinuousLitModule,
#         datamodule_class=MultiSourceTrainDatasetModule,
#         seed_everything_default=42,
#         save_config_overwrite=True,
#         run=False,
#         auto_registry=True,
#         parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
#     )
#
#     trainer = cli.trainer
#
#     os.makedirs(trainer.default_root_dir, exist_ok=True)
#
#     cli.datamodule.set_patch_size(cli.model.get_patch_size())
#
#     cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())
#
#     # fit() runs the training
#     ckpt_path = os.path.join(trainer.default_root_dir, 'checkpoints', 'last.ckpt')
#     print (ckpt_path)
#     if (not os.path.exists(ckpt_path)) or (cli.model.pretrained_path != ''):
#         ckpt_path = None
#         print ('No ckpt found')
#     trainer.fit(cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_path)
