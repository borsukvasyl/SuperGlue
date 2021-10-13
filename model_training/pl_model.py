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
        self.loss = nn.NLLLoss()

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        out = self.model(batch["kpts0"], batch["descs0"], batch["kpts1"], batch["descs1"])

        torch.log_softmax(out, dim=2)
        loss = self.loss(out, targets)

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

    def _parse_keypoints(self, kpts, scores, descriptors):
        kpts = torch.cat([kpts, scores[..., None]], dim=2)
        return kpts.permute(0, 2, 1), descriptors.permute(0, 2, 1)
