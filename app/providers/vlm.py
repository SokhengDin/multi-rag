import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from app.core.config import settings
from app.core.device import check_device, check_dtype, check_quantization

class LocalVLM:

    def __init__(self, model_id: str, bits : int | None = None) -> None:
        self.device    = check_device()
        self.model_id  = model_id

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype               = check_dtype(self.device),
            quantization_config = check_quantization(self.device, bits),
            device_map          = self.device,
        )
        self.model.eval()

    @torch.inference_mode()
    def generate(self, messages: list[dict], max_new_tokens: int = 512) -> str:
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt = True,
            tokenize              = True,
            return_dict           = True,
            return_tensors        = "pt",
        ).to(self.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        new_ids    = output_ids[0, inputs["input_ids"].shape[-1]:]

        return self.processor.decode(new_ids, skip_special_tokens=True).strip()


class MLXVLM:

    def __init__(self, model_id: str) -> None:
        from mlx_vlm       import load
        from mlx_vlm.utils import load_config

        self.model_id              = model_id
        self.model, self.processor = load(model_id)
        self.config                = load_config(model_id)

    def generate(self, messages: list[dict], max_new_tokens: int = 512) -> str:
        from mlx_vlm              import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        images, text_messages = _split_images(messages)

        prompt = apply_chat_template(self.processor, self.config, text_messages, num_images=len(images))
        output = generate(self.model, self.processor, prompt, image=images or None, max_tokens=max_new_tokens, verbose=False)

        return getattr(output, "text", output).strip()

def _split_images(messages: list[dict]) -> tuple[list, list[dict]]:
    
    images        = []
    text_messages = []

    for message in messages:
        content = message["content"]

        if isinstance(content, str):
            text_messages.append(message)
            continue

        texts = []
        for part in content:
            if part["type"] == "image":
                images.append(part["image"])
            elif part["type"] == "text":
                texts.append(part["text"])

        text_messages.append({"role": message["role"], "content": "\n".join(texts)})

    return images, text_messages


def load_vlm() -> LocalVLM | MLXVLM:
    if check_device() == "mlx":
        return MLXVLM(settings.VLM_MODEL_ID)
    return LocalVLM(settings.VLM_MODEL_ID, settings.QUANT_BITS)

