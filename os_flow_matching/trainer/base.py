from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch import nn

from ..dataset.sampleable import Sampleable
from ..dataset.transform import Transform
from ..utils import ResultsManager, WandbManager
from ..utils.metrics import resolve_metrics
from .contracts import EvalResult, DataEvalRequest, SplitEvalRequest, LossDict
from .evaluation import EvaluationMixin
from .train_loop import TrainingLoopMixin
from .transfer import TransferEvaluationMixin


class Trainer(TrainingLoopMixin, EvaluationMixin, TransferEvaluationMixin, ABC):
    def __init__(
        self,
        model: nn.Module,
        results_manager: ResultsManager,
        wandb_manager: WandbManager | None = None,
        experiment_name: str | None = None,
        tracked_metrics: tuple[str, ...] | None = None,
        early_stopping_patience: int = 0,
        condition_on_x0: bool = False,
        x0_noise_std: float = 0.0,
        grad_clip_threshold: float = 10.0,
    ):
        super().__init__()
        self.model = model
        self.results_manager = results_manager
        self.wandb_manager = wandb_manager
        self.experiment_name = experiment_name or "experiment"
        self.condition_on_x0 = condition_on_x0
        self.x0_noise_std = x0_noise_std

        self.grad_clip_threshold = grad_clip_threshold

        self.tracked_metrics = tuple(
            tracked_metrics
            or ("ht_intrinsic_mean", "l2_mean", "dtw_mean")
        )
        self.metrics = resolve_metrics(self.tracked_metrics)
        self._best_model_records: dict[str, dict[str, Any]] = {
            name: {"value": float("inf"), "epoch": None}
            for name in self.tracked_metrics
        }

        self.early_stopping_patience = int(early_stopping_patience)
        self._stalled_validations = 0
        self._stopped_epoch: int | None = None

    @abstractmethod
    def get_train_loss(self, split: str) -> LossDict:
        raise NotImplementedError

    @abstractmethod
    def get_train_loss_at(self, indices: list[int], split: str) -> LossDict:
        raise NotImplementedError

    @property
    @abstractmethod
    def transform(self) -> Transform:
        raise NotImplementedError

    @abstractmethod
    def _get_num_batches(self, split: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def _evaluate_split(self, request: SplitEvalRequest) -> EvalResult:
        raise NotImplementedError

    @abstractmethod
    def _evaluate_indices(
        self,
        indices: list[int],
        request: SplitEvalRequest,
    ) -> EvalResult:
        raise NotImplementedError

    @abstractmethod
    def _evaluate_data(self, request: DataEvalRequest) -> EvalResult:
        raise NotImplementedError

    @abstractmethod
    def _get_sampler_and_datasets(self) -> tuple[Sampleable, dict[str, Any]]:
        raise NotImplementedError

    @property
    def best_models(self) -> dict[str, dict[str, Any]]:
        return {
            name: {"value": record["value"], "epoch": record["epoch"]}
            for name, record in self._best_model_records.items()
        }

    @staticmethod
    def _ensure_no_unexpected_kwargs(kwargs: dict[str, Any], context: str) -> None:
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword arguments for {context}: {unexpected}")


__all__ = ["Trainer"]
