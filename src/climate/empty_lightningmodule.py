from typing import Any

from pytorch_lightning import LightningModule

from src.utils.registry import registry


@registry.register_lightningmodule(name="emptymodule")
class EmptyModule(LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        # do something  naive here
        # the reason to use this lighting shell to load batch to GPU is for ease of testing multi-node setup, which
        # is easily setup ny lightning
        return
