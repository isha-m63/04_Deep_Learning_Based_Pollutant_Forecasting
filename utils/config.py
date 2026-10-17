import os
import yaml

DEFAULT_CONFIG_PATH = os.path.join("configs", "config.yaml")


def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load the YAML config. Paths inside it are relative to the repository root."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
