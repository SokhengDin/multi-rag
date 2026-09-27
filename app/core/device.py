import importlib.util
import torch

from transformers import BitsAndBytesConfig, QuantoConfig

from app.core.config import settings


def is_mlx_available() -> bool:
    """MLX is a separate Apple Silicon framework, not a torch backend."""
    if importlib.util.find_spec("mlx") is None:
        return False
    import mlx.core as mx

    return mx.metal.is_available()


def check_device() -> str:
    """Return settings.DEVICE if available, or the best one when "auto": cuda > mps > cpu."""
    requested = settings.DEVICE.lower()
    available = available_devices()

    if requested == "auto":
        if available["cuda"]:
            return "cuda"
        if available["mps"]:
            return "mps"
        return "cpu"

    if requested not in available:
        raise ValueError(f"DEVICE={settings.DEVICE!r} is invalid, use one of: auto, {', '.join(available)}")
    if not available[requested]:
        raise RuntimeError(f"DEVICE={requested} was requested but is not available on this machine")

    return requested

def check_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def available_devices() -> dict[str, bool]:
    return {
        "cuda": torch.cuda.is_available(),
        "mps" : torch.backends.mps.is_available(),
        "mlx" : is_mlx_available(),
        "cpu" : True,
    }


def check_quantization(device: str, bits: int | None):
    """bitsandbytes on cuda, quanto on mps/cpu. MLX quantizes its own way, so it never gets here."""
    if bits is None:
        return None

    if bits not in (4, 8):
        raise ValueError(f"QUANT_BITS={bits} is invalid, use 4, 8, or leave it empty")

    if device == "cuda":
        _require("bitsandbytes", "bitsandbytes")
        return BitsAndBytesConfig(
            load_in_4bit    = bits == 4,
            load_in_8bit    = bits == 8,
            bnb_4bit_compute_dtype = check_dtype(device)
        )

    _require("optimum.quanto", "optimum-quanto")
    return QuantoConfig(weights="int4" if bits == 4 else "int8")


def _require(module: str, package: str) -> None:
    if importlib.util.find_spec(module.split(".")[0]) is None or importlib.util.find_spec(module) is None:
        raise ImportError(f"QUANT_BITS is set but {package} is not installed, run: uv add {package}")