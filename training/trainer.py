"""One training loop for every model (the models differ only in how inputs are fed and in an optional regulariser)."""
import os
import timeit
from typing import Callable, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from models.registry import ModelSpec, get_spec


def fit(spec: ModelSpec, model: nn.Module, train_loader, test_loader, device, epochs: int = 50, lr: float = 1e-3,
        save_path: Optional[str] = None, losses_save_path: Optional[str] = None,
        extra_loss: Optional[Callable] = None):
    """MSE training with Adam. `extra_loss(model, inputs, outputs)` is added to the training loss only."""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses = [], []

    start_time = timeit.default_timer()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if targets.ndim > 2:
                targets = targets.squeeze()

            optimizer.zero_grad()
            outputs = spec.forward(model, inputs)
            loss = criterion(outputs, targets)
            if extra_loss is not None:
                loss = loss + extra_loss(model, inputs, outputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        avg_train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                if targets.ndim > 2:
                    targets = targets.squeeze()
                val_loss += criterion(spec.forward(model, inputs), targets).item() * inputs.size(0)
        avg_val_loss = val_loss / len(test_loader.dataset)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    total_time = timeit.default_timer() - start_time
    print(f"Total training time: {total_time:.2f} seconds / {total_time/60:.2f} minutes")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model weights saved to {save_path}")

    if losses_save_path:
        os.makedirs(losses_save_path, exist_ok=True)
        out = os.path.join(losses_save_path, f"{spec.key}_loss_epoch.csv")
        pd.DataFrame({"epoch": range(1, epochs + 1), "train_loss": train_losses, "val_loss": val_losses}).to_csv(out, index=False)
        print(f"Train and validation losses saved to {out}")

    return train_losses, val_losses


def main(key: str) -> None:
    """Command-line entry point shared by training/train_<model>.py:  python -m training.train_lstm --horizon 3"""
    import argparse
    from utils.config import horizon_suffix, load_config, with_horizon
    from utils.data_loader import get_data_loaders

    spec = get_spec(key)
    parser = argparse.ArgumentParser(description=f"Train the {spec.label} model")
    parser.add_argument("--horizon", type=int, default=0, help="hours ahead to predict (0 = nowcast)")
    config = with_horizon(load_config(), parser.parse_args().horizon)
    sfx, t = horizon_suffix(config), config["training"]

    torch.manual_seed(42)   # reproducible runs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_data_loaders(config, batch_size=t["batch_size"])

    loss_dir = config["evaluation"]["loss_epoch_path"][f"{key}_losses"] + (f"h{config['horizon']}/" if sfx else "")
    fit(spec, spec.build(config["model"][key]), train_loader, test_loader, device,
        epochs=t["epochs"], lr=t["learning_rate"],
        save_path=config["weight_paths"][spec.weights_key].replace(".pth", sfx + ".pth"),
        losses_save_path=loss_dir,
        extra_loss=spec.make_extra_loss(t) if spec.make_extra_loss else None)
