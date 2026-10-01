import torch


def softmax_ref(x: torch.Tensor) -> torch.Tensor:
    """Ground truth: row-wise softmax over the last dim of a 2D tensor."""
    return torch.softmax(x, dim=-1)
