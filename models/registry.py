"""
Single place that knows how each model is built from the config and how a window of inputs is fed to it.

Training, evaluation and the Streamlit app all go through this registry, so adding a model means adding one
`ModelSpec` here (open for extension) instead of editing an if/elif in every consumer.
"""
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn

from models.lstm import LSTMRegressor
from models.neural_ode import NeuralODERegressor
from models.pinn import PINNRegressor

NODE_TIME_GRID = torch.linspace(0, 1, 10)      # integration grid of the neural ODE (same as in training)


def last_step(x: torch.Tensor) -> torch.Tensor:
    """(batch, seq_len, features) -> (batch, features): models that use only the latest hour."""
    return x[:, -1, :] if x.ndim == 3 else x


def _pinn_smoothness(training_cfg: dict) -> Callable:
    """Soft temporal-smoothness penalty: lambda * mean(((f(x_t) - f(x_{t-1})) / dt)^2)."""
    lambda_phys, delta_t = training_cfg.get("pinn_lambda_phys", 1.0), training_cfg.get("pinn_delta_t", 1.0)

    def extra_loss(model: nn.Module, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            return torch.zeros((), device=outputs.device)
        outputs_prev = model(inputs[:, -2, :])
        return lambda_phys * torch.mean(((outputs - outputs_prev) / delta_t) ** 2)

    return extra_loss


@dataclass(frozen=True)
class ModelSpec:
    key: str                                                    # name in config['model'] and in file names
    label: str                                                  # display name
    build: Callable[[dict], nn.Module]                          # model config -> untrained module
    forward: Callable[[nn.Module, torch.Tensor], torch.Tensor]  # (module, windows) -> (batch, 3)
    weights_key: str                                            # key in config['weight_paths']
    make_extra_loss: Optional[Callable[[dict], Callable]] = None  # training-time regulariser, if any


REGISTRY: dict[str, ModelSpec] = {
    "lstm": ModelSpec(
        "lstm", "LSTM",
        lambda c: LSTMRegressor(c["input_size"], c["hidden_size"], c["num_layers"], c["output_size"], c["dropout"]),
        lambda m, x: m(x),
        "lstm_model",
    ),
    "neural_ode": ModelSpec(
        "neural_ode", "Neural ODE",
        lambda c: NeuralODERegressor(c["input_size"], c["hidden_size"], c["intermediate_size"], c["output_size"]),
        lambda m, x: m(last_step(x), NODE_TIME_GRID.to(x.device)),
        "neural_ode_model",
    ),
    "pinn": ModelSpec(
        "pinn", "PINN",
        lambda c: PINNRegressor(c["input_size"], c["hidden_size"], c["output_size"], c["num_layers"]),
        lambda m, x: m(last_step(x)),
        "pinn_model",
        _pinn_smoothness,
    ),
}


def get_spec(key: str) -> ModelSpec:
    try:
        return REGISTRY[key]
    except KeyError:
        raise KeyError(f"Unknown model '{key}'. Available: {sorted(REGISTRY)}") from None


def load_model(spec: ModelSpec, config: dict, weights_path, device="cpu") -> nn.Module:
    """Build the model from config['model'][spec.key] and load a state dict into it (eval mode)."""
    model = spec.build(config["model"][spec.key]).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    return model.eval()
