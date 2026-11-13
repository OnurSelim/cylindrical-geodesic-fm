from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from tqdm import tqdm

from ..dataset.sampleable import Sampleable
from ..dataset.transform import Transform
from ..utils.general import crop_center
from ..utils.metrics import compute_all_metrics, resolve_metrics
from .base import Trainer
from .common import _format_metric_summary
from .contracts import (
    EvalResult,
    DataEvalRequest,
    PredictionHistory,
    SplitEvalRequest,
    LossDict,
    make_eval_result,
    make_loss_dict,
)


class SupervisedTrainer(Trainer):
    def __init__(
        self,
        sampler: Sampleable,
        **kwargs,
    ):
        self._sampler = sampler
        super().__init__(
            **kwargs,
        )

    @property
    def transform(self) -> Transform:
        return self._sampler.transform

    @property
    def sampler(self) -> Sampleable:
        return self._sampler

    def get_train_loss(self, split: str) -> LossDict:
        src, cond, tgt = self._sampler.sample(split=split)
        if tgt is None:
            raise ValueError("tgt must not be None")
        return self._compute_supervised_loss(src, cond, tgt, split)

    def get_train_loss_at(self, indices: list[int], split: str) -> LossDict:
        src, cond, tgt = self._sampler.sample_at(indices, split=split)
        if tgt is None:
            raise ValueError("tgt must not be None")
        return self._compute_supervised_loss(src, cond, tgt, split)

    def _get_sampler_and_datasets(self) -> tuple[Sampleable, dict[str, Any]]:
        return self.sampler, self.sampler.datasets

    def _get_num_batches(self, split: str) -> int:
        return (self.sampler.dataset_size(split) + self.sampler.batch_size - 1) // self.sampler.batch_size

    def _prepare_x0_conditioning(self, src: Tensor, split: str) -> Tensor | None:
        x0_cond = src if self.condition_on_x0 else None
        if x0_cond is not None and self.x0_noise_std > 0 and split == "train":
            x0_cond = x0_cond + self.x0_noise_std * torch.randn_like(x0_cond)
        return x0_cond

    def _forward_supervised(self, src: Tensor, cond: Tensor | None, split: str) -> Tensor:
        t = torch.ones(src.shape[0], 1, 1, device=src.device)
        x0_cond = self._prepare_x0_conditioning(src, split)
        return self.model(src, t, cond, x0_cond)

    def _compute_supervised_loss(
        self,
        src: Tensor,
        cond: Tensor | None,
        tgt: Tensor,
        split: str,
    ) -> LossDict:
        pred = self._forward_supervised(src, cond, split)
        diff = pred - tgt
        loss = diff.reshape(diff.shape[0], -1).pow(2).sum(dim=1).mean()
        return make_loss_dict(
            loss=loss,
            log_values={"total_loss": float(loss.detach().item())},
        )

    @torch.no_grad()
    def _evaluate_split(self, request: SplitEvalRequest) -> EvalResult:
        self.model.to(request["device"])
        self.model.eval()

        num_batches = self._get_num_batches(request["split"])
        all_pred: list[Tensor] = []
        all_target: list[Tensor] = []

        for _ in tqdm(range(num_batches), desc=f"{request['desc_prefix']} on {request['split']}", leave=False):
            src, cond, tgt = self._sampler.sample(split=request["split"])
            src = src.to(request["device"])
            if cond is not None:
                cond = cond.to(request["device"])
            if tgt is not None:
                tgt = tgt.to(request["device"])
            if tgt is None:
                raise ValueError("tgt must not be None")

            pred = self._forward_supervised(src, cond, request["split"])
            all_pred.append(pred)
            all_target.append(tgt)

        pred_all = torch.cat(all_pred, dim=0)
        target_all = torch.cat(all_target, dim=0)

        metrics = compute_all_metrics(pred=pred_all, target=target_all, metrics=self.metrics)
        metrics.update({"num_samples": pred_all.size(0)})

        history = None
        if request["return_history"]:
            history = {
                "pred_traj": pred_all.unsqueeze(1),
                "t_values": torch.ones(1, device=pred_all.device),
                "target_final": target_all,
            }
        return make_eval_result(metrics=metrics, history=history)

    @torch.no_grad()
    def _evaluate_indices(
        self,
        indices: list[int],
        request: SplitEvalRequest,
    ) -> EvalResult:
        self.model.to(request["device"])
        self.model.eval()

        all_pred: list[Tensor] = []
        all_target: list[Tensor] = []

        for batch_indices in self._iter_index_batches(indices, self.sampler.batch_size, shuffle=False):
            src, cond, tgt = self._sampler.sample_at(batch_indices, split=request["split"])
            src = src.to(request["device"])
            if cond is not None:
                cond = cond.to(request["device"])
            if tgt is not None:
                tgt = tgt.to(request["device"])
            if tgt is None:
                raise ValueError("tgt must not be None")

            pred = self._forward_supervised(src, cond, request["split"])
            all_pred.append(pred)
            all_target.append(tgt)

        pred_all = torch.cat(all_pred, dim=0)
        target_all = torch.cat(all_target, dim=0)

        metrics = compute_all_metrics(pred=pred_all, target=target_all, metrics=self.metrics)
        metrics.update({"num_samples": pred_all.size(0)})
        return make_eval_result(metrics=metrics)

    @torch.no_grad()
    def _evaluate_data(self, request: DataEvalRequest) -> EvalResult:
        batch_size = self.sampler.batch_size
        num_samples = request["x0_all"].shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        all_pred: list[Tensor] = []

        for i in tqdm(range(num_batches), desc=f"  Running {self.experiment_name}", leave=False):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)

            x0_batch = self.transform.crop_to_model_size(request["x0_all"][start_idx:end_idx]).to(request["device"])
            cond_batch = None
            if request["cond_all"] is not None:
                cond_batch = self.transform.crop_to_model_size(
                    request["cond_all"][start_idx:end_idx]
                ).to(request["device"])

            t = torch.ones(x0_batch.shape[0], 1, 1, device=request["device"])
            x0_cond = x0_batch if self.condition_on_x0 else None
            all_pred.append(self.model(x0_batch, t, cond_batch, x0_cond))

        x1_pred = torch.cat(all_pred, dim=0)
        crop_size = request["model_window_size"] if request["model_window_size"] is not None else x1_pred.shape[-1]

        x0_cropped = (
            crop_center(request["x0_all"], crop_size)
            if request["x0_all"].shape[-1] != crop_size
            else request["x0_all"]
        ).to(request["device"])
        x1_pred = crop_center(x1_pred, crop_size) if x1_pred.shape[-1] != crop_size else x1_pred
        x1_target_cropped = (
            crop_center(request["x1_all"], crop_size)
            if request["x1_all"].shape[-1] != crop_size
            else request["x1_all"]
        ).to(request["device"])

        metrics = compute_all_metrics(
            pred=x1_pred,
            target=x1_target_cropped,
            metrics=resolve_metrics(request["metric_names"]),
        )
        print(f"  Test metrics: {_format_metric_summary(request['metric_names'], metrics)}")

        history: PredictionHistory = {
            "pred_traj": torch.stack([x0_cropped, x1_pred], dim=1),
            "t_values": torch.tensor([0.0, 1.0], device=request["device"]),
            "target_final": x1_target_cropped,
        }
        return make_eval_result(metrics=metrics, history=history)


__all__ = ["SupervisedTrainer"]
