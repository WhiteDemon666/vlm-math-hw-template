from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import numpy as np
from PIL import Image

from hw.constants import IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN, IGNORE_INDEX
from hw.dataset import MathVQASample


@dataclass
class ProcessorConfig:
    image_size: int = 224
    num_tiles: int = 1
    tile_overlap: float = 0.0
    num_image_tokens: int = 49
    max_length: int = 512
    ignore_index: int = IGNORE_INDEX


class MathVLMProcessor:
    """Builds model inputs from MathVQASample.

    The processor owns all text/image preprocessing that must be deterministic
    across train and inference.
    """

    def __init__(self, tokenizer: Any, config: ProcessorConfig | None = None) -> None:
        self.tokenizer = tokenizer
        self.config = config or ProcessorConfig()

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Convert image to tensor with shape [num_tiles, 3, image_size, image_size].
        """
        image = image.resize((self.config.image_size, self.config.image_size))

        img_arr = np.array(image)
        tensor = torch.tensor(img_arr).permute(2, 0, 1).float()
        tensor = tensor / 255.0

        return tensor.unsqueeze(0)

    def build_prompt(self, sample: MathVQASample, include_answer: bool) -> str:
        """Build a text prompt with visual special tokens and options.

        For training, include_answer=True should append the assistant answer.
        For inference, include_answer=False should stop before the answer.
        """
        img_tokens = IMAGE_TOKEN * self.config.num_image_tokens
        pref = f"{IMAGE_START_TOKEN}{img_tokens}{IMAGE_END_TOKEN}\n"

        text = "\n".join(sample.options) if sample.options else ""
        prompt = f"{pref}Question: {sample.question}\nOptions:\n{text}\nAnswer:"
        if include_answer:
            prompt += f" {sample.answer}"

        return prompt

    def tokenize_sample(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        """Return input_ids, attention_mask and labels for one sample.

        labels must be IGNORE_INDEX for prompt tokens and real token ids only
        for the assistant answer.
        """
        full_text = self.build_prompt(sample, include_answer=True)
        prompt_text = self.build_prompt(sample, include_answer=False)

        full_token = self.tokenizer(full_text, truncation=True,
                                    max_length=self.config.max_length)
        prompt_token = self.tokenizer(prompt_text, truncation=True,
                                      max_length=self.config.max_length)

        input_ids = torch.tensor(full_token["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(full_token["attention_mask"], dtype=torch.long)
        prompt_length = len(prompt_token["input_ids"])

        labels = input_ids.clone()
        labels[:prompt_length] = self.config.ignore_index

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def __call__(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        item = self.tokenize_sample(sample)
        item["pixel_values"] = self.preprocess_image(sample.image)
        return item

    def collate(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Pad text fields and stack pixel_values.
        """
        processed_batch = []
        for item in batch:
            if isinstance(item, MathVQASample):
                processed_batch.append(self(item))
            else:
                processed_batch.append(item)

        input_ids = [item["input_ids"] for item in processed_batch]
        attention_mask = [item["attention_mask"] for item in processed_batch]
        labels = [item["labels"] for item in processed_batch]
        pixel_values = [item["pixel_values"] for item in processed_batch]

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True,
                                                           padding_value=pad_id)
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True,
                                                                padding_value=0)
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True,
                                                        padding_value=self.config.ignore_index)

        pixel_values_stacked = torch.stack(pixel_values, dim=0)

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded,
            "pixel_values": pixel_values_stacked
        }
