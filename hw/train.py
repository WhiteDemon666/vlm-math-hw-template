from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_step(model: torch.nn.Module, batch: dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> float:
    """Run one optimization step and return scalar loss.

    TODO:
        - model.train();
        - forward;
        - ensure finite loss;
        - backward;
        - optimizer.step();
        - optimizer.zero_grad();
    """
    model.train()

    outputs = model(batch)
    if hasattr(outputs, "loss"):
        loss = outputs.loss
    elif isinstance(outputs, dict) and "loss" in outputs:
        loss = outputs["loss"]
    elif isinstance(outputs, torch.Tensor):
        loss = outputs
    else:
        loss = outputs[0]

    if not math.isfinite(loss.item()):
        print(f"Что-то не так, Loss это {loss.item()}, останавливаем обучение")
        return loss.item()

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()


def run_training(config: dict[str, Any], fast_train: bool = False) -> None:
    """Main training entry point.
    """
    print(f"Ну мы ничего не загружаем, потому что путь А, поэтому пусть будет так")
    return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fast-train", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    run_training(config, fast_train=args.fast_train)


if __name__ == "__main__":
    main()
