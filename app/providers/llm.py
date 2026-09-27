import re

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from app.core.config import settings
from app.core.device import check_dtype, check_device, check_quantization


class LocalLLM:

    def __init__(
        self,
        model_id : str,
        bits     : int | None = None
    ) -> None:
        self.device     = check_device()
        self.model_id   = model_id

        self.tokenizer  = AutoTokenizer.from_pretrained(model_id)
        self.model      = AutoModelForCausalLM.from_pretrained(
            model_id
            , dtype = check_dtype(self.device)
            , quantization_config = check_quantization(self.device, bits)
            , device_map = self.device
        )
        self.model.eval()

    @torch.inference_mode()
    def generate(self, messages: list[dict], max_new_tokens: int = 512) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True,
            tokenize    = True,
            return_dict = True,
            return_tensors = "pt"
        ).to(self.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        new_ids    = output_ids[0, inputs["input_ids"].shape[-1]:]

        return strip_thinking(self.tokenizer.decode(new_ids, skip_special_tokens=True))



class MLXLLM:
    def __init__(
        self,
        model_id: str
    ) -> None:

        from mlx_lm import load

        self.model_id   = model_id
        self.model, self.tokenizer = load(model_id)

    def generate(self, messages: list[dict], max_new_tokens: int = 512) -> str:
        from mlx_lm import generate
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        output = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_new_tokens)

        return strip_thinking(output)


def strip_thinking(text: str) -> str:
    """Remove the <think>...</think> block that reasoning models (e.g. Qwen3) put before the answer."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def load_llm() -> LocalLLM | MLXLLM:
    if check_device() == "mlx":
        return MLXLLM(settings.LLM_MODEL_ID)
    return LocalLLM(settings.LLM_MODEL_ID, settings.QUANT_BITS)

    