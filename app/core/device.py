import importlib.util

import torch


def is_mlx_available() -> bool:
    """MLX is a separate Apple Silicon framework, not a torch backend."""
    if importlib.util.find_spec("mlx") is None:
        return False
    import mlx.core as mx

    return mx.metal.is_available()


def check_device() -> str:
    """Return the best available torch device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def available_devices() -> dict[str, bool]:
    return {
        "cuda": torch.cuda.is_available(),
        "mps" : torch.backends.mps.is_available(),
        "mlx" : is_mlx_available(),
        "cpu" : True,
    }
