from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from tqdm import tqdm

from settings import FS

from ..model import ConditionalVectorField
from ..path import (
    ConditionalProbabilityPath,
)
from ..dataset.sampleable import Sampleable
from ..dataset.transform import Transform
from ..simulators import ProjectedHeunSimulator, UNetVectorFieldODE
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


class VanillaFMTrainer(Trainer):
    def __init__(
        self,
        path: ConditionalProbabilityPath,
        model: ConditionalVectorField,
        tracked_metrics: tuple[str, ...] | None = None,
        psd_ratio_loss_weight: float = 0.0,
        psd_ratio_band_hz: tuple[float, float] = (1.0, 39.0),
        training_num_steps: int | None = None,
        psd_step_size: float | None = None,
        channel_weights: tuple[float, ...] | None = None,
        **kwargs,
    ):
        super().__init__(model=model, tracked_metrics=tracked_metrics, **kwargs)
        self.path = path
        self.channel_weights = channel_weights
        self.psd_ratio_loss_weight = psd_ratio_loss_weight
        self.fs_hz = float(FS)
        self.psd_ratio_band_hz = psd_ratio_band_hz
        self.training_num_steps = None if training_num_steps is None else int(training_num_steps)
        self.psd_step_size = None if psd_step_size is None else float(psd_step_size)
        if self.psd_step_size is not None and self.psd_step_size <= 0:
            raise ValueError(
                f"psd_step_size must be positive when set, got {self.psd_step_size}.",
            )
        self._simulator = ProjectedHeunSimulator(path=self.path)

    @property
    def transform(self) -> Transform:
        return self.path.transform

    def get_train_loss(self, split: str) -> LossDict:
        (xt, x0, x1), t, cond = self.path.sample_conditional_path(split=split)
        return self._compute_loss(xt, x0, x1, t, cond, split)

    def get_train_loss_at(self, indices: list[int], split: str) -> LossDict:
        (xt, x0, x1), t, cond = self._sample_conditional_path_at(indices, split)
        return self._compute_loss(xt, x0, x1, t, cond, split)

    def _get_sampler_and_datasets(self) -> tuple[Sampleable, dict[str, Any]]:
        sampler = self.path.sampler
        return sampler, sampler.datasets

    def _get_num_batches(self, split: str) -> int:
        sampler, _ = self._get_sampler_and_datasets()
        return (sampler.dataset_size(split) + sampler.batch_size - 1) // sampler.batch_size

    def _sample_conditional_path_at(
        self,
        indices: list[int],
        split: str,
    ) -> tuple[tuple[Tensor, Tensor, Tensor], Tensor, Tensor | None]:
        x0, cond, x1 = self.path.sampler.sample_at(indices, split=split)
        return self.path.compute_conditional_path(
            x0=x0,
            x1=x1,
            cond=cond,
            already_transformed=True,
        )

    def _simulate_batch(
        self,
        x0: Tensor,
        cond: Tensor | None,
        base_ts: Tensor,
        num_steps: int,
        return_history: bool = False,
        num_intermediate: int | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        ode = UNetVectorFieldODE(self.model)
        simulator = ProjectedHeunSimulator(ode, path=self.path)

        x0_cond = x0 if self.condition_on_x0 else None
        batch_size = x0.size(0)
        extra_dims = x0.dim() - 1
        ts = base_ts.view(1, num_steps, *([1] * extra_dims)).expand(batch_size, -1, *([1] * extra_dims))

        if return_history:
            pred_traj = simulator.simulate_with_trajectory(
                x0,
                ts,
                cond=cond,
                x0=x0_cond,
                num_intermediate=num_intermediate,
            )
            return pred_traj[:, -1], pred_traj
        return simulator.simulate(x0, ts, cond=cond, x0=x0_cond)

    @torch.no_grad()
    def _evaluate_split(self, request: SplitEvalRequest) -> EvalResult:
        self.model.to(request["device"])
        self.model.eval()

        num_batches = self._get_num_batches(request["split"])
        all_x1_pred: list[Tensor] = []
        all_x1_target: list[Tensor] = []
        all_pred_traj: list[Tensor] = []
        base_ts = torch.linspace(0, 1, request["num_steps"], device=request["device"])

        for _ in tqdm(range(num_batches), desc=f"{request['desc_prefix']} on {request['split']}", leave=False):
            (_, x0, x1_target), _, cond = self.path.sample_conditional_path(split=request["split"])

            x0 = self.path.to_path_coords(x0.to(request["device"]))
            x1_target = self.path.to_path_coords(x1_target.to(request["device"]))
            if cond is not None:
                cond = self.path.to_path_coords(cond.to(request["device"]))

            result = self._simulate_batch(
                x0=x0,
                cond=cond,
                base_ts=base_ts,
                num_steps=request["num_steps"],
                return_history=request["return_history"],
            )
            if request["return_history"]:
                x1_pred, pred_traj = result
                all_pred_traj.append(pred_traj)
            else:
                x1_pred = result

            all_x1_pred.append(x1_pred)
            all_x1_target.append(x1_target)

        x1_pred_all = torch.cat(all_x1_pred, dim=0)
        x1_target_all = torch.cat(all_x1_target, dim=0)

        x1_pred_ts = self.path.to_time_series(x1_pred_all)
        x1_target_ts = self.path.to_time_series(x1_target_all)

        metrics = compute_all_metrics(pred=x1_pred_ts, target=x1_target_ts, metrics=self.metrics)
        metrics.update({"num_samples": x1_pred_ts.size(0), "num_steps": request["num_steps"]})

        history: PredictionHistory | None = None
        if request["return_history"]:
            pred_traj = (
                torch.cat(all_pred_traj, dim=0)
                if all_pred_traj
                else torch.empty(0, request["num_steps"], device=request["device"])
            )
            pred_traj_ts = self.path.to_time_series(pred_traj)
            history = {
                "pred_traj": pred_traj_ts,
                "t_values": base_ts,
                "target_final": x1_target_ts,
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

        sampler = self.path.sampler
        all_x1_pred: list[Tensor] = []
        all_x1_target: list[Tensor] = []
        base_ts = torch.linspace(0, 1, request["num_steps"], device=request["device"])

        for batch_indices in self._iter_index_batches(indices, sampler.batch_size, shuffle=False):
            x0, cond, x1 = sampler.sample_at(batch_indices, split=request["split"])

            x0 = self.path.to_path_coords(x0.to(request["device"]))
            x1_target = self.path.to_path_coords(x1.to(request["device"]))

            if cond is not None and cond.numel() > 0:
                cond = self.path.to_path_coords(cond.to(request["device"]))
            else:
                cond = None

            x1_pred = self._simulate_batch(
                x0=x0,
                cond=cond,
                base_ts=base_ts,
                num_steps=request["num_steps"],
                return_history=False,
            )
            all_x1_pred.append(x1_pred)
            all_x1_target.append(x1_target)

        x1_pred_all = torch.cat(all_x1_pred, dim=0)
        x1_target_all = torch.cat(all_x1_target, dim=0)

        x1_pred_ts = self.path.to_time_series(x1_pred_all)
        x1_target_ts = self.path.to_time_series(x1_target_all)

        metrics = compute_all_metrics(pred=x1_pred_ts, target=x1_target_ts, metrics=self.metrics)
        metrics.update({"num_samples": x1_pred_ts.size(0), "num_steps": request["num_steps"]})
        return make_eval_result(metrics=metrics)

    @torch.no_grad()
    def _evaluate_data(self, request: DataEvalRequest) -> EvalResult:
        batch_size = self._get_sampler_and_datasets()[0].batch_size
        num_samples = request["x0_all"].shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        base_ts = torch.linspace(0, 1, request["num_steps"], device=request["device"])

        all_x1_pred: list[Tensor] = []
        all_pred_traj: list[Tensor] = []

        for i in tqdm(range(num_batches), desc=f"  Running {self.experiment_name}", leave=False):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)

            x0_batch = request["x0_all"][start_idx:end_idx].to(request["device"])
            x1_batch = request["x1_all"][start_idx:end_idx].to(request["device"])
            cond_batch = None
            if request["cond_all"] is not None:
                cond_batch = request["cond_all"][start_idx:end_idx].to(request["device"])

            t_zero = torch.zeros(x0_batch.size(0), 1, device=request["device"], dtype=x0_batch.dtype)
            (_, x0_tf, _), _, cond_tf = self.path.compute_conditional_path(
                x0=x0_batch,
                x1=x1_batch,
                cond=cond_batch,
                t=t_zero,
                already_transformed=True,
            )
            x0_path = self.path.to_path_coords(x0_tf)
            cond_path = self.path.to_path_coords(cond_tf) if cond_tf is not None and cond_tf.numel() > 0 else cond_tf

            x1_pred_batch, pred_traj_batch = self._simulate_batch(
                x0=x0_path,
                cond=cond_path,
                base_ts=base_ts,
                num_steps=request["num_steps"],
                return_history=True,
                num_intermediate=20,
            )
            all_x1_pred.append(x1_pred_batch)
            all_pred_traj.append(pred_traj_batch)

        x1_pred_cat = torch.cat(all_x1_pred, dim=0)
        pred_traj_cat = torch.cat(all_pred_traj, dim=0)

        x1_pred = self.path.to_time_series(x1_pred_cat)
        pred_traj_ts = self.path.to_time_series(pred_traj_cat)

        crop_size = request["model_window_size"] if request["model_window_size"] is not None else x1_pred.shape[-1]
        if pred_traj_ts.shape[-1] != crop_size:
            raise ValueError(
                f"Prediction trajectory length ({pred_traj_ts.shape[-1]}) "
                f"does not match crop size ({crop_size})!"
            )

        x1_pred = crop_center(x1_pred, crop_size) if x1_pred.shape[-1] != crop_size else x1_pred
        x1_target_ts = self.path.transform.backward(request["x1_all"].to(request["device"]))

        metrics = compute_all_metrics(
            pred=x1_pred,
            target=x1_target_ts,
            metrics=resolve_metrics(request["metric_names"]),
        )
        print(f"  Test metrics: {_format_metric_summary(request['metric_names'], metrics)}")

        history: PredictionHistory = {
            "pred_traj": pred_traj_ts,
            "t_values": base_ts,
            "target_final": x1_target_ts,
        }
        return make_eval_result(metrics=metrics, history=history)


    def _step_dt(self, t: Tensor) -> Tensor:
        if self.psd_step_size is None:
            return (1.0 - t).clamp_min(0.0)

        t_like = t
        base_dt = torch.full_like(t_like, self.psd_step_size)
        remaining = (1.0 - t_like).clamp_min(0.0)
        return torch.minimum(base_dt, remaining)

    def _compute_psd_ratio_loss(self, signal: Tensor) -> Tensor:
        num_samples = signal.shape[-1]
        spectrum = torch.fft.rfft(signal, dim=-1)
        psd = spectrum.abs().pow(2) / float(num_samples)

        freqs = torch.fft.rfftfreq(num_samples, d=1.0 / self.fs_hz).to(signal.device)
        lo_hz, hi_hz = self.psd_ratio_band_hz
        band_mask = (freqs >= lo_hz) & (freqs <= hi_hz)
        band_power = psd[..., band_mask].sum(dim=-1)
        total_power = psd.sum(dim=-1)
        eps = torch.finfo(psd.dtype).eps
        ratio = band_power / total_power.clamp_min(eps)
        ratio_per_sample = ratio.reshape(ratio.shape[0], -1).mean(dim=1)
        return (1.0 - ratio_per_sample).mean()

    def _compute_oracle_vector_field(
        self,
        x0: Tensor,
        xt: Tensor,
        x1: Tensor,
        t: Tensor,
        cond: Tensor | None,
    ) -> Tensor:
        return self.path.conditional_vector_field(x0=x0, xt=xt, x1=x1, t=t, cond=cond)

    def _prepare_x0_conditioning(self, x0: Tensor, split: str) -> Tensor | None:
        x0_cond = x0 if self.condition_on_x0 else None
        if x0_cond is not None and self.x0_noise_std > 0 and split == "train":
            x0_cond = x0_cond + self.x0_noise_std * torch.randn_like(x0_cond)
        return x0_cond

    def _compute_fm_loss(
        self,
        ut_ref: Tensor,
        ut_theta: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        diff = ut_theta - ut_ref
        sq_error = diff.pow(2)

        # Per-channel losses (before weighting) — only for multi-channel VFs
        _CHANNEL_NAMES = {
            2: ("amp", "phase"),
            3: ("vx", "vy", "amp"),
        }
        channel_logs: dict[str, float] = {}
        if self.channel_weights is not None:
            n_channels = sq_error.shape[-1]
            names = _CHANNEL_NAMES.get(n_channels, tuple(f"ch{i}" for i in range(n_channels)))
            for ch, name in enumerate(names):
                ch_loss = sq_error[..., ch].reshape(sq_error.shape[0], -1).sum(dim=1).mean()
                channel_logs[f"fm_loss_{name}"] = float(ch_loss.detach().item())

        if self.channel_weights is not None:
            w = sq_error.new_tensor(self.channel_weights)
            sq_error = sq_error * w

        total = sq_error.reshape(sq_error.shape[0], -1).sum(dim=1).mean()
        return total, channel_logs

    def _compute_psd_loss(
        self,
        xt: Tensor,
        t: Tensor,
        ut_theta: Tensor,
    ) -> Tensor:
        dt = self._step_dt(t)
        xt_next = self._simulator.step(
            xt=xt,
            t=t,
            dt=dt,
            drift=ut_theta,
        )
        xt_next_ts = self.path.to_time_series(xt_next)
        return self._compute_psd_ratio_loss(xt_next_ts)

    def _compose_total_loss(
        self,
        fm_loss: Tensor,
        psd_loss: Tensor | None,
        channel_logs: dict[str, float] | None = None,
    ) -> LossDict:
        total_loss = fm_loss

        # PSD loss
        psd_loss_weighted_val = 0.0
        if psd_loss is not None and self.psd_ratio_loss_weight > 0:
            eps = torch.finfo(fm_loss.dtype).eps
            psd_scale = fm_loss.detach() / psd_loss.detach().clamp_min(eps)
            alpha = self.psd_ratio_loss_weight
            beta = alpha / max(1.0 - alpha, eps)
            psd_loss_weighted = beta * psd_loss * psd_scale
            total_loss = total_loss + psd_loss_weighted
            psd_loss_weighted_val = float(psd_loss_weighted.detach().item())

        log_values = {
            "fm_loss": float(fm_loss.detach().item()),
            "psd_loss": float(psd_loss.detach().item()) if psd_loss is not None else 0.0,
            "psd_loss_weighted": psd_loss_weighted_val,
            "total_loss": float(total_loss.detach().item()),
        }
        if channel_logs:
            log_values.update(channel_logs)
        return make_loss_dict(loss=total_loss, log_values=log_values)

    def _compute_loss(
        self,
        xt: Tensor,
        x0: Tensor,
        x1: Tensor,
        t: Tensor,
        cond: Tensor | None,
        split: str,
    ) -> LossDict:
        # xt, x0, x1, cond are in transform space.
        # VF operates in transform space.
        ut_ref = self._compute_oracle_vector_field(x0=x0, xt=xt, x1=x1, t=t, cond=cond)

        # Convert to path coords for model and simulator.
        xt_p = self.path.to_path_coords(xt)
        cond_p = self.path.to_path_coords(cond) if cond is not None and cond.numel() > 0 else cond
        x0_path = self.path.to_path_coords(x0)
        x0_cond = self._prepare_x0_conditioning(x0_path, split)
        ut_theta = self.model(xt_p, t, cond_p, x0_cond)

        fm_loss, channel_logs = self._compute_fm_loss(ut_ref=ut_ref, ut_theta=ut_theta)

        psd_loss = None
        if self.psd_ratio_loss_weight > 0:
            psd_loss = self._compute_psd_loss(xt=xt_p, t=t, ut_theta=ut_theta)

        return self._compose_total_loss(
            fm_loss=fm_loss, psd_loss=psd_loss, channel_logs=channel_logs,
        )


__all__ = ["VanillaFMTrainer"]
