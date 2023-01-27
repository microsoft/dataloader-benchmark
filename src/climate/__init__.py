from pytorch_lightning import LightningModule

from src.utils.registry import registry


def _try_register_climate_modules():
    try:
        from .pretrain_multi_source_module import MultiSourceTrainDatasetModule

        print("import climate modules success")
    except ImportError as e:
        climate_import_error = e

        print("import multiMAE module failed", e)

        @registry.register_lightningmodule(name="ClimateErr")
        class ClimateImportError(LightningModule):
            def __init__(self, *args, **kwargs):
                raise climate_import_error
