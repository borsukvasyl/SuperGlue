import logging
import os

import yaml


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
