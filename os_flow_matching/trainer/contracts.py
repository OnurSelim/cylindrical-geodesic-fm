from __future__ import annotations

from typing import Any, Mapping, TypedDict

import torch
from torch import Tensor


LogValue = float | Tensor


class PredictionHistory(TypedDict):
    pred_traj: Tensor
    t_values: Tensor
    target_final: Tensor


class LossDict(TypedDict):
    loss: Tensor
    log_values: dict[str, LogValue]


class EvalResult(TypedDict):
    metrics: dict[str, Any]
    history: PredictionHistory | None


class SplitEvalRequest(TypedDict):
    split: str
    device: torch.device
    desc_prefix: str
    num_steps: int
    return_history: bool


class DataEvalRequest(TypedDict):
    x0_all: Tensor
    x1_all: Tensor
    cond_all: Tensor | None
    device: torch.device
    num_steps: int
    metric_names: tuple[str, ...]
    model_window_size: int | None


def make_loss_dict(
    loss: Tensor,
    log_values: Mapping[str, LogValue] | None = None,
) -> LossDict:
    return {
        "loss": loss,
        "log_values": dict(log_values or {}),
    }


def make_eval_result(
    metrics: Mapping[str, Any],
    history: PredictionHistory | None = None,
) -> EvalResult:
    return {
        "metrics": dict(metrics),
        "history": history,
    }


def make_split_eval_request(
    *,
    split: str,
    device: torch.device,
    desc_prefix: str,
    num_steps: int,
    return_history: bool,
) -> SplitEvalRequest:
    return {
        "split": split,
        "device": device,
        "desc_prefix": desc_prefix,
        "num_steps": int(num_steps),
        "return_history": bool(return_history),
    }


def make_fixed_data_eval_request(
    *,
    x0_all: Tensor,
    x1_all: Tensor,
    cond_all: Tensor | None,
    device: torch.device,
    num_steps: int,
    metric_names: tuple[str, ...],
    model_window_size: int | None,
) -> DataEvalRequest:
    return {
        "x0_all": x0_all,
        "x1_all": x1_all,
        "cond_all": cond_all,
        "device": device,
        "num_steps": int(num_steps),
        "metric_names": tuple(metric_names),
        "model_window_size": model_window_size,
    }


__all__ = [
    "EvalResult",
    "DataEvalRequest",
    "LogValue",
    "PredictionHistory",
    "SplitEvalRequest",
    "LossDict",
    "make_eval_result",
    "make_fixed_data_eval_request",
    "make_split_eval_request",
    "make_loss_dict",
]
