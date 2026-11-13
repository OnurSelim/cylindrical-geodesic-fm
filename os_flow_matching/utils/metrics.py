from __future__ import annotations

import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable

import numpy as np
import torch
from dtaidistance import dtw

from .hilbert import ht_transform


def circ_diff(
    x1: torch.Tensor,
    x2: torch.Tensor,
    *,
    period: float | torch.Tensor,
) -> torch.Tensor:
    p = torch.as_tensor(period, dtype=x1.dtype, device=x1.device)
    half_p = 0.5 * p
    delta = x1 - x2
    wrapped = torch.remainder(delta + half_p, p) - half_p
    return wrapped.abs()


class Metric(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def produces(self) -> tuple[str, ...]:
        """All metric keys produced by this metric (e.g. ('l2_mean', 'l2_var'))."""
        pass

    @abstractmethod
    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        pass


class L2Metric(Metric):
    @property
    def name(self) -> str:
        return "l2"

    @property
    def produces(self) -> tuple[str, ...]:
        return ("l2_mean", "l2_var")

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        diff2 = (pred - target).pow(2)
        mse_per_beat = diff2.view(diff2.size(0), -1).mean(dim=1)
        rmse_per_beat = mse_per_beat.sqrt()
        return {
            "l2_mean": rmse_per_beat.mean().item(),
            "l2_var": rmse_per_beat.var(unbiased=False).item(),
        }


class DTWMetric(Metric):
    def __init__(self, downsample_factor: int = 4):
        self.downsample_factor = downsample_factor

    @property
    def name(self) -> str:
        return "dtw"

    @property
    def produces(self) -> tuple[str, ...]:
        return ("dtw_mean", "dtw_std")

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        pred_flat = pred.detach().cpu().reshape(pred.size(0), -1).numpy()
        target_flat = target.detach().cpu().reshape(target.size(0), -1).numpy()

        pred_flat = np.ascontiguousarray(pred_flat, dtype=np.double)
        target_flat = np.ascontiguousarray(target_flat, dtype=np.double)

        if self.downsample_factor is not None and self.downsample_factor > 1:
            pred_flat = self._downsample(pred_flat, self.downsample_factor)
            target_flat = self._downsample(target_flat, self.downsample_factor)

        num_samples = pred_flat.shape[0]
        indices = list(range(num_samples))
        dist_fn = dtw.distance_fast

        def _pair_dtw(i: int) -> float:
            return dist_fn(pred_flat[i], target_flat[i])

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            dtw_distances = list(executor.map(_pair_dtw, indices))

        dtw_arr = np.asarray(dtw_distances, dtype=float)

        return {
            "dtw_mean": float(np.mean(dtw_arr)),
            "dtw_std": float(np.std(dtw_arr)),
        }

    @staticmethod
    def _downsample(arr: np.ndarray, factor: int) -> np.ndarray:
        T = arr.shape[1]
        L = (T // factor) * factor
        if L == 0:
            return arr
        arr = arr[:, :L]
        arr = arr.reshape(arr.shape[0], -1, factor).mean(axis=2)
        return np.ascontiguousarray(arr, dtype=np.double)


class HTMetric(Metric):
    def __init__(self, fs: float = 500.0, eps: float = 1e-12):
        self.fs = fs
        self.eps = eps

    @property
    def name(self) -> str:
        return "ht"

    @property
    def produces(self) -> tuple[str, ...]:
        return ("ht_intrinsic_mean", "ht_intrinsic_std")

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)

        amp_pred, phi_pred, _ = ht_transform(pred, fs=self.fs, eps=self.eps)
        amp_tgt, phi_tgt, _ = ht_transform(target, fs=self.fs, eps=self.eps)

        intrinsic_dist = self._compute_intrinsic_distance(
            amp_pred, phi_pred, amp_tgt, phi_tgt
        )
        intrinsic_per_beat = intrinsic_dist.view(intrinsic_dist.size(0), -1).mean(dim=1)

        return {
            "ht_intrinsic_mean": intrinsic_per_beat.mean().item(),
            "ht_intrinsic_std": intrinsic_per_beat.std(unbiased=False).item(),
        }

    def _compute_intrinsic_distance(
        self,
        amp1: torch.Tensor,
        phi1: torch.Tensor,
        amp2: torch.Tensor,
        phi2: torch.Tensor,
    ) -> torch.Tensor:
        du = amp1 - amp2
        dphi = circ_diff(phi1, phi2, period=2.0 * torch.pi)
        return torch.sqrt(dphi.pow(2) + du.pow(2))


DEFAULT_METRICS: list[Metric] = [
    L2Metric(),
    DTWMetric(downsample_factor=4),
    HTMetric(),
]


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    metrics: list[Metric] | None = None,
) -> Dict[str, float]:
    if metrics is None:
        metrics = DEFAULT_METRICS

    results: Dict[str, float] = {}
    for metric in metrics:
        results.update(metric.compute(pred, target))
    return results


def _build_metric_key_registry(metrics: Iterable[Metric]) -> dict[str, Metric]:
    reg: dict[str, Metric] = {}
    for m in metrics:
        for k in m.produces:
            if k in reg:
                raise ValueError(f"Duplicate metric key {k!r} produced by {m!r} and {reg[k]!r}")
            reg[k] = m
    return reg


_METRIC_BY_KEY: dict[str, Metric] = _build_metric_key_registry(DEFAULT_METRICS)


def resolve_metrics(metric_keys: Iterable[str]) -> list[Metric]:
    wanted = set(metric_keys)
    unknown = sorted(wanted - set(_METRIC_BY_KEY.keys()))
    if unknown:
        known = ", ".join(sorted(_METRIC_BY_KEY.keys()))
        raise KeyError(f"Unknown metric(s): {unknown}. Known: {known}")

    needed = {_METRIC_BY_KEY[k] for k in wanted}
    return [m for m in DEFAULT_METRICS if m in needed]