import os
import yaml

DEFAULT_CONFIG_PATH = os.path.join("configs", "config.yaml")


def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load the YAML config. Paths inside it are relative to the repository root."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def horizon_suffix(config: dict) -> str:
    """"" for the nowcast, "_h3" for a 3-hour-ahead forecast; used in weight / result file names."""
    h = int(config.get("horizon", 0))
    return f"_h{h}" if h else ""


def with_horizon(config: dict, horizon: int) -> dict:
    return {**config, "horizon": int(horizon)}
