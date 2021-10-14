import torch
from fire import Fire

from model_training.model import SuperGlue
from model_training.utils import load_yaml


def main(config_path: str, jit_save_path: str):
    config = load_yaml(config_path)
    model = SuperGlue(**config["model"])
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, jit_save_path)


if __name__ == '__main__':
    Fire(main)
