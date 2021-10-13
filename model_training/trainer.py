import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def _get_logger(config):
    save_dir = os.path.join(config["experiment_path"], "logs")
    return [TensorBoardLogger(save_dir=save_dir)]


def _get_checkpoint_callback(config):
    dirpath = os.path.join(config["experiment_path"], "checkpoints")
    return ModelCheckpoint(
        dirpath=dirpath,
        monitor=config.get("metric_to_monitor", "loss"),
        mode=config.get("metric_mode", "min"),
        save_top_k=3,
        save_last=config.get("save_last", True),
        verbose=True,
    )


def get_trainer(config):
    checkpoint = _get_checkpoint_callback(config)
    trainer = Trainer(
        logger=_get_logger(config),
        gpus=config["gpus"],
        precision=config.get("precision", 32),
        callbacks=[checkpoint],
        min_epochs=config["min_epochs"],
        max_epochs=config["max_epochs"],
        val_check_interval=config.get("val_check_interval", 1.0),
        limit_train_batches=config.get("train_percent", 1.0),
        limit_val_batches=config.get("val_percent", 1.0),
        progress_bar_refresh_rate=config.get("progress_bar_refresh_rate", 10),
        num_sanity_val_steps=config.get("sanity_steps", 5),
        log_every_n_steps=1,
    )
    return trainer
