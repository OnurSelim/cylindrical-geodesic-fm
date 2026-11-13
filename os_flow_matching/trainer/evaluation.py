from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from tqdm import tqdm

from ..utils.metrics import compute_all_metrics
from .common import _history_to_dict
from .contracts import (
    EvalResult,
    SplitEvalRequest,
    make_split_eval_request,
    make_fixed_data_eval_request,
)


class EvaluationMixin:
    @torch.no_grad()
    def compute_baseline_metrics(
        self,
        device: torch.device,
        split: str = "val",
    ) -> dict[str, Any]:
        num_batches = self._get_num_batches(split)
        sampler, _ = self._get_sampler_and_datasets()
        transform = self.transform
        all_source: list[Tensor] = []
        all_target: list[Tensor] = []

        for _ in tqdm(range(num_batches), desc=f"Computing baseline on {split}", leave=False):
            source, _, target = sampler.sample(split=split)
            source = transform.backward(source)
            target = transform.backward(target)
            all_source.append(source.to(device))
            all_target.append(target.to(device))

        all_source = torch.cat(all_source, dim=0)
        all_target = torch.cat(all_target, dim=0)

        metrics = compute_all_metrics(pred=all_source, target=all_target, metrics=self.metrics)
        metrics.update({"num_samples": all_source.size(0), "epoch": 0})
        return metrics

    def evaluate_data(
        self,
        x0_all: Tensor,
        x1_all: Tensor,
        cond_all: Tensor | None,
        device: torch.device,
        *,
        num_steps: int,
        metric_name: str,
        metric_names: tuple[str, ...],
        model_window_size: int | None,
        fewshot_experiment: str | None = None,
        fewshot_checkpoint_type: str = "best",
    ) -> tuple[dict[str, Any] | None, dict[str, float]]:
        result = self._evaluate_data_inputs(
            x0_all=x0_all,
            x1_all=x1_all,
            cond_all=cond_all,
            device=device,
            num_steps=num_steps,
            metric_name=metric_name,
            metric_names=metric_names,
            model_window_size=model_window_size,
            fewshot_experiment=fewshot_experiment,
            fewshot_checkpoint_type=fewshot_checkpoint_type,
        )
        return _history_to_dict(result["history"]), dict(result["metrics"])

    def load_checkpoint(
        self,
        metric_name: str,
        *,
        device: torch.device,
    ) -> dict[str, Any]:
        return self.results_manager.load_checkpoint(
            run_id=self.experiment_name,
            model=self.model,
            metric_name=metric_name,
            device=device,
        )

    def _build_split_eval_request(
        self,
        *,
        split: str,
        device: torch.device,
        desc_prefix: str,
        kwargs: dict[str, Any],
    ) -> SplitEvalRequest:
        options = dict(kwargs)
        num_steps = int(options.pop("num_steps", 40))
        return_history = bool(options.pop("return_history", False))
        self._ensure_no_unexpected_kwargs(options, desc_prefix.lower())
        return make_split_eval_request(
            split=split,
            device=device,
            desc_prefix=desc_prefix,
            num_steps=num_steps,
            return_history=return_history,
        )

    def _evaluate_data_inputs(
        self,
        *,
        x0_all: Tensor,
        x1_all: Tensor,
        cond_all: Tensor | None,
        device: torch.device,
        num_steps: int,
        metric_name: str,
        metric_names: tuple[str, ...],
        model_window_size: int | None,
        fewshot_experiment: str | None = None,
        fewshot_checkpoint_type: str = "best",
    ) -> EvalResult:
        request = make_fixed_data_eval_request(
            x0_all=x0_all,
            x1_all=x1_all,
            cond_all=cond_all,
            device=device,
            num_steps=num_steps,
            metric_names=metric_names,
            model_window_size=model_window_size,
        )
        self._load_eval_checkpoint(
            metric_name=metric_name,
            device=device,
            fewshot_experiment=fewshot_experiment,
            fewshot_checkpoint_type=fewshot_checkpoint_type,
        )
        self.model.to(device)
        self.model.eval()
        return self._evaluate_data(request)

    def _load_eval_checkpoint(
        self,
        *,
        metric_name: str,
        device: torch.device,
        fewshot_experiment: str | None,
        fewshot_checkpoint_type: str,
    ) -> dict[str, Any]:
        if fewshot_experiment is None:
            checkpoint = self.results_manager.load_checkpoint(
                run_id=self.experiment_name,
                model=self.model,
                metric_name=metric_name,
                device=device,
            )
            print(
                f"Loaded checkpoint for {metric_name} from epoch {checkpoint['epoch']} "
                f"(metric value: {checkpoint['metric_value']:.6f})"
            )
            return checkpoint

        checkpoint = self.results_manager.load_fewshot_checkpoint(
            run_id=self.experiment_name,
            model=self.model,
            metric_name=metric_name,
            experiment=fewshot_experiment,
            device=device,
            checkpoint_type=fewshot_checkpoint_type,
        )
        print(
            f"  Loaded few-shot {fewshot_checkpoint_type} checkpoint ({fewshot_experiment}): "
            f"epoch {checkpoint['epoch']}, {metric_name}={checkpoint['metric_value']:.6f}"
        )
        return checkpoint

__all__ = ["EvaluationMixin"]
