import logging
import os

import torch
import yaml
import torch.nn as nn


def create_logger(name: str) -> logging.Logger:
    logger = logging.Logger(name)
    console_handler = logging.StreamHandler()
    level = logging.DEBUG if os.environ.get("DEBUG") else logging.INFO
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    return logger


def load_yaml(x: str):
    with open(x) as fd:
        config = yaml.load(fd, yaml.FullLoader)
        config["yaml_path"] = x
        return config


def get_relative_path(path: str, relative_to: str) -> str:
    return os.path.join(os.path.dirname(relative_to), path)


def load_from_lighting(model: nn.Module, checkpoint_path: str, map_location=None) -> nn.Module:
    map_location = f"cuda:{map_location}" if type(map_location) is int else map_location
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = {
        k.lstrip("model").lstrip("."): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")
    }
    model.load_state_dict(state_dict)
    return model
