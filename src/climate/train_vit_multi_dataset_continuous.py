import os

from pytorch_lightning.utilities.cli import LightningCLI

from models.vit_continuous_module import ViTContinuousLitModule
from src.datamodules.pretrain_multi_source_module import \
    MultiSourceTrainDatasetModule


def main():
    # Initialize Lightning with the model and data modules, and instruct it to parse the config yml
    cli = LightningCLI(
        model_class=ViTContinuousLitModule,
        datamodule_class=MultiSourceTrainDatasetModule,
        seed_everything_default=42,
        save_config_overwrite=True,
        run=False,
        auto_registry=True,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )

    trainer = cli.trainer

    os.makedirs(trainer.default_root_dir, exist_ok=True)

    cli.datamodule.set_patch_size(cli.model.get_patch_size())
    
    cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())

    # fit() runs the training
    ckpt_path = os.path.join(trainer.default_root_dir, 'checkpoints', 'last.ckpt')
    print (ckpt_path)
    if (not os.path.exists(ckpt_path)) or (cli.model.pretrained_path != ''):
        ckpt_path = None
        print ('No ckpt found')
    trainer.fit(cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
