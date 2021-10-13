import torch
import pytorch_lightning as pl
import torch.nn as nn
from pytorch_toolbelt.optimization.functional import get_optimizable_parameters
from torch import optim


class SuperGlueLightningModel(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.config = config
        self.model = model
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        kpts0 = batch["kpts0"]
        desc0 = batch["desc0"]
        kpts1 = batch["kpts1"]
        desc1 = batch["desc1"]
        matches = batch["matches"]
        predicted_matches = self.model(kpts0, desc0, kpts1, desc1)

        loss = self.loss(predicted_matches, matches)

        # logging
        self.log('train/loss', loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # get optimizer
        optimizer_config = self.config["optimizer"]
        params = get_optimizable_parameters(self.model)
        optimizer = torch.optim.Adam(params, lr=optimizer_config.get("lr", 1e-4))

        # get scheduler
        scheduler_config = self.config["scheduler"]
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.config.get("metric_mode", "min"),
            patience=scheduler_config["patience"],
            factor=scheduler_config["factor"],
            min_lr=scheduler_config["min_lr"],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": scheduler_config.get("metric_to_monitor", "train/loss"),
        }
