from fire import Fire

from model_training.data import SuperGlueDataset
from model_training.pl_model import SuperGlueLightningModel
from model_training.trainer import get_trainer
from model_training.utils import load_yaml
from superglue.model import SuperGlue


def train_superglue(config):
    model = SuperGlue(**config["model"])
    pl_model = SuperGlueLightningModel(model=model, config=config)
    dataset = SuperGlueDataset.from_config(config["data"])
    trainer = get_trainer(config)
    trainer.fit(pl_model, dataset)


def main(config_path: str):
    config = load_yaml(config_path)
    train_superglue(config)


if __name__ == '__main__':
    Fire(main)
