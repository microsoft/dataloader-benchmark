from typing import Any

import torch.nn as nn
from pytorch_lightning import LightningModule

from src.utils.registry import registry


@registry.register_lightningmodule(name="emptymodule")
class EmptyModule(LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = nn.Linear(3, 4)

    def training_step(self, batch, batch_idx):
        # do something  naive here
        # the reason to use this lighting shell to load batch to GPU is for ease of testing multi-node setup, which
        # is easily setup ny lightning
        print("batch loaded")
        return

    def configure_optimizers(self) -> Any:
        return None
