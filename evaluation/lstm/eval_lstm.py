import torch

from evaluation.common import evaluate, main
from models.registry import get_spec


def evaluate_model(model_path, config, device):
    return evaluate(get_spec("lstm"), model_path, config, device)


if __name__ == "__main__":
    main("lstm")
