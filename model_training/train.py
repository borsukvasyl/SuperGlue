from fire import Fire

from model_training.dataset import SuperGlueDataset
from model_training.lightning.pl_model import SuperGlueLightningModel
from model_training.lightning.trainer import get_trainer
from model_training.model import SuperGlue
from model_training.utils import load_yaml, load_from_lighting, create_logger

logger = create_logger("Train")


def get_model(config):
    model_config = config["model"]
    model = SuperGlue(**model_config)
    if config.get("checkpoint", ""):
        checkpoint = config["checkpoint"]
        logger.info(f"Loading checkpoint: {checkpoint}")
        load_from_lighting(model, checkpoint)
    return model


def train_superglue(config):
    model = get_model(config)
    dataset = SuperGlueDataset(**config["data"])
    pl_model = SuperGlueLightningModel(model=model, dataset=dataset, config=config)
    trainer = get_trainer(config)
    trainer.fit(pl_model)


def main(config_path: str):
    config = load_yaml(config_path)
    train_superglue(config)


if __name__ == '__main__':
    Fire(main)
