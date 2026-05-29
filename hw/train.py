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
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPVisionConfig, CLIPVisionModel
    from hw.dataset import MathVQADataset
    from hw.processor import MathVLMProcessor, ProcessorConfig
    from hw.model import MathVLM, ModelConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"

    manifest_path = config.get("manifest_path", "assets/toy_math_vqa/manifest.jsonl")
    dataset = MathVQADataset(manifest_path, split="train", max_samples=4 if fast_train else None)

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceM4/tiny-random-LlamaForCausalLM")
    language_model = AutoModelForCausalLM.from_pretrained("HuggingFaceM4/tiny-random-LlamaForCausalLM")

    vision_config = CLIPVisionConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, patch_size=16)
    vision_encoder = CLIPVisionModel(vision_config)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.add_tokens(["<image>"])
    language_model.resize_token_embeddings(len(tokenizer))
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")

    processor = MathVLMProcessor(tokenizer, ProcessorConfig())
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=processor.collate)

    model_config = ModelConfig(
        vision_hidden_size=vision_encoder.config.hidden_size,
        text_hidden_size=language_model.config.hidden_size,
        num_image_tokens=processor.config.num_image_tokens,
        image_token_id=image_token_id
    )

    model = MathVLM(vision_encoder, language_model, model_config).to(device)
    model.freeze_backbones()

    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=1e-4)
    max_steps = 2 if fast_train else 10
    step = 0
    total_loss = 0.0

    print("Обучаем")
    for batch in dataloader:
        if step >= max_steps:
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        loss = train_one_step(model, batch, optimizer)
        total_loss += loss
        print(f"Эпоха {step + 1}/{max_steps} | Loss: {loss:.4f}")
        step += 1

    print(f"Обучили. Средний Loss: {total_loss / max(1, step):.4f}\n")
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
