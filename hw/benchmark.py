from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

from hw.constants import CHOICES


def normalize_text(text: str) -> str:
    """Simple normalization for free-form answers."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_mc_answer(text: str, choices: tuple[str, ...] = CHOICES) -> str | None:
    """Extract multiple-choice answer letter from model output.

    TODO:
        Handle cases like:
            "A"
            "(B)"
            "Answer: C"
            "The correct answer is D."
    """
    text = text.strip()

    match = re.search(r'(?:answer\s*is\s*|answer:\s*)\(?([A-D])\)?', text, re.IGNORECASE)
    if match:
        ans = match.group(1).upper()
        if ans in choices:
            return ans

    match = re.search(r'\(([A-D])\)', text, re.IGNORECASE)
    if match:
        ans = match.group(1).upper()
        if ans in choices:
            return ans

    matches = re.findall(r'\b([A-D])\b', text, re.IGNORECASE)
    if matches:
        ans = matches[-1].upper()
        if ans in choices:
            return ans

    return


def build_benchmark_prompt(question: str, options: list[str]) -> str:
    """Build prompt for multiple-choice visual math evaluation."""
    options_text = "\n".join(options)
    return (
        "Реши визуально-математическую задачу. "
        "Выбери один вариант ответа и в конце напиши только букву.\n\n"
        f"Вопрос: {question}\n"
        f"Варианты:\n{options_text}\n"
        "Ответ:"
    )


def compute_accuracy(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute overall and per-subject accuracy from prediction rows."""
    if not rows:
        return {"overall": 0.0}

    total = len(rows)
    correct = sum(int(r.get("prediction") == r.get("answer")) for r in rows)
    metrics = {"overall": correct / total}

    subjects = sorted({r.get("subject", "unknown") for r in rows})
    for subject in subjects:
        sub_rows = [r for r in rows if r.get("subject", "unknown") == subject]
        sub_correct = sum(int(r.get("prediction") == r.get("answer")) for r in sub_rows)
        metrics[f"subject/{subject}"] = sub_correct / max(1, len(sub_rows))
    return metrics


def run_benchmark(config: dict[str, Any], toy: bool = False) -> dict[str, float]:
    """Run evaluation loop.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPVisionConfig, CLIPVisionModel
    from hw.dataset import MathVQADataset
    from hw.processor import MathVLMProcessor, ProcessorConfig
    from hw.model import MathVLM, ModelConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"

    manifest_path = config.get("eval_manifest_path", "assets/toy_math_vqa/manifest.jsonl")
    dataset = MathVQADataset(manifest_path, split="val", max_samples=4 if toy else None)
    if len(dataset) == 0:
        dataset = MathVQADataset(manifest_path, split="train", max_samples=4 if toy else None)

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceM4/tiny-random-LlamaForCausalLM")
    language_model = AutoModelForCausalLM.from_pretrained("HuggingFaceM4/tiny-random-LlamaForCausalLM")

    vision_config = CLIPVisionConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, patch_size=16)
    vision_encoder = CLIPVisionModel(vision_config)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.add_tokens(["<image>"])
    language_model.resize_token_embeddings(len(tokenizer))

    processor = MathVLMProcessor(tokenizer, ProcessorConfig())
    model_config = ModelConfig(
        vision_hidden_size=vision_encoder.config.hidden_size,
        text_hidden_size=language_model.config.hidden_size,
        num_image_tokens=processor.config.num_image_tokens,
        image_token_id=tokenizer.convert_tokens_to_ids("<image>")
    )

    model = MathVLM(vision_encoder, language_model, model_config).to(device)
    model.eval()

    results = []
    print("Отвечаем")
    for i in range(len(dataset)):
        sample = dataset[i]

        batch_single = processor.collate([sample])
        batch_single = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_single.items()}

        with torch.no_grad():
            out_ids = model.generate(batch_single, max_new_tokens=5)

        gen_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        prediction = parse_mc_answer(gen_text)

        results.append({"id": sample.id, "subject": sample.subject, "answer": sample.answer, "prediction": prediction})

    metrics = compute_accuracy(results)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--toy", action="store_true")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    metrics = run_benchmark(config, toy=args.toy)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
