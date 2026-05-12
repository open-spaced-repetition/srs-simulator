from __future__ import annotations


def enable_expandable_cuda_segments() -> bool:
    """Enable expandable CUDA allocator segments when supported by PyTorch."""
    try:
        import torch
    except ImportError:
        return False

    try:
        set_allocator_settings = getattr(
            torch.cuda.memory,
            "_set_allocator_settings",
            None,
        )
    except AttributeError:
        return False
    if set_allocator_settings is None:
        return False

    try:
        set_allocator_settings("expandable_segments:True")
    except (AttributeError, RuntimeError, TypeError):
        return False
    return True


__all__: list[str] = ["enable_expandable_cuda_segments"]
