from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ..utils.general import crop_center


def _to_float(value: float | Tensor) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def _history_to_dict(history: dict[str, Any] | None) -> dict[str, Any] | None:
    if history is None:
        return None
    return dict(history)


def _crop_center(x: Tensor | None, size: int | None) -> Tensor | None:
    if x is None or size is None:
        return x
    return crop_center(x, size)


def _format_metric_summary(metric_names: tuple[str, ...], metrics: dict[str, Any]) -> str:
    parts: list[str] = []
    for name in metric_names:
        value = metrics.get(name)
        if value is None:
            parts.append(f"{name}=N/A")
        else:
            parts.append(f"{name}={float(value):.6f}")
    return ", ".join(parts)


__all__ = [
    "_crop_center",
    "_format_metric_summary",
    "_history_to_dict",
    "_to_float",
]
