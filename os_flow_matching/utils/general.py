from __future__ import annotations

import torch
from torch import nn as nn

MiB = 1024 ** 2


def crop_center(x: torch.Tensor, size: int) -> torch.Tensor:
    if size <= 0:
        raise ValueError("size must be a positive integer")
    if x.shape[-1] == size:
        return x
    if size > x.shape[-1]:
        raise ValueError(f"size ({size}) exceeds input length ({x.shape[-1]})")
    crop_amount = (x.shape[-1] - size) // 2
    return x[..., crop_amount:crop_amount + size]


def model_size_b(model: nn.Module) -> int:
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size
