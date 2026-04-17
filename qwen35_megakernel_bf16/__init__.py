"""Qwen3.5 Megakernel BF16 - Custom CUDA kernel implementation."""

try:
    from .qwen35_megakernel_bf16_C import megakernel_fn
except ImportError as e:
    raise ImportError(
        "Failed to import qwen35_megakernel_bf16_C. "
        "Ensure the CUDA extension is built correctly. "
        f"Original error: {e}"
    ) from e

__all__ = ["megakernel_fn"]
__version__ = "0.0.0"
