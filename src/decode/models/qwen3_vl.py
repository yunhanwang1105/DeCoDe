from __future__ import annotations

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from decode.config import ModelConfig
from decode.models.base import MultimodalLLM

try:
    from qwen_vl_utils import process_vision_info
except ImportError as exc:
    raise ImportError(
        "qwen-vl-utils is required for Qwen3-VL vision inputs. "
        "Install dependencies with: pip install -r requirements.txt"
    ) from exc


class Qwen3VLAdapter(MultimodalLLM):
    """Qwen3-VL implementation of the DeCoDe model interface."""

    def __init__(self, config: ModelConfig) -> None:
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.model_name_or_path,
            torch_dtype=config.torch_dtype,
            device_map=config.device_map,
            cache_dir=str(config.cache_dir),
        )
        self.processor = AutoProcessor.from_pretrained(
            config.model_name_or_path,
            cache_dir=str(config.cache_dir),
        )
        self.processor.tokenizer.padding_side = "left"
        self.yes_token_id = self.processor.tokenizer.encode(
            "Yes",
            add_special_tokens=False,
        )[0]

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def score_yes(self, messages: list[dict], max_new_tokens: int) -> float:
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False,
            )

        logits = outputs.scores[0][0]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return float(log_probs[self.yes_token_id].item())

    def score_yes_batch(self, messages_batch: list[list[dict]], max_new_tokens: int) -> list[float]:
        texts = [
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in messages_batch
        ]
        image_inputs, video_inputs = process_vision_info(messages_batch)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False,
            )

        first_step_logits = outputs.scores[0]
        log_probs = torch.nn.functional.log_softmax(first_step_logits, dim=-1)
        return log_probs[:, self.yes_token_id].detach().cpu().tolist()
