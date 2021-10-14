from fire import Fire

from model_training.dataset import SuperGlueDataset
from model_training.pl_model import SuperGlueLightningModel
from model_training.trainer import get_trainer
from model_training.utils import load_yaml
from model_training.model import SuperGlue


def train_superglue(config):
    model = SuperGlue(**config["model"])
    dataset = SuperGlueDataset(**config["data"])
    pl_model = SuperGlueLightningModel(model=model, dataset=dataset, config=config)
    trainer = get_trainer(config)
    trainer.fit(pl_model)


def main(config_path: str):
    config = load_yaml(config_path)
    train_superglue(config)


if __name__ == '__main__':
    Fire(main)
