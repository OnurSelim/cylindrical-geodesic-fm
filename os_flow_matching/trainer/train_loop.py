from __future__ import annotations

from typing import Any

import torch
from tqdm import tqdm

from ..utils import ExperimentConfig
from ..utils.general import MiB, model_size_b
from .common import _to_float


class TrainingLoopMixin:
    def train(
        self,
        config: ExperimentConfig,
        device: torch.device,
        split: str = "train",
        **kwargs,
    ) -> tuple[list[float], list[dict[str, Any]]]:
        self._ensure_no_unexpected_kwargs(dict(kwargs), "train")
        optimizer, scheduler, steps_per_epoch = self._prepare_training(
            config=config,
            device=device,
            split=split,
        )

        loss_arr: list[float] = []
        val_metrics_arr: list[dict[str, Any]] = []

        for epoch in range(int(config.num_epochs)):
            avg_loss, avg_components = self._run_training_epoch(
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                num_epochs=int(config.num_epochs),
                split=split,
                steps_per_epoch=steps_per_epoch,
            )
            current_lr = optimizer.param_groups[0]["lr"]
            self._log_training_epoch(
                epoch=epoch,
                num_epochs=int(config.num_epochs),
                avg_loss=avg_loss,
                avg_components=avg_components,
                current_lr=current_lr,
            )
            loss_arr.append(avg_loss)
            if self._run_validation(epoch, config, device, val_metrics_arr):
                self._stopped_epoch = epoch + 1
                print(
                    f"Early stopping at epoch {epoch + 1}/{int(config.num_epochs)}: "
                    f"no improvement in any tracked metric for "
                    f"{self._stalled_validations} validations"
                )
                break

        self._finalize_training(loss_arr, val_metrics_arr, config)
        return loss_arr, val_metrics_arr

    def get_optimizer(
        self,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.99),
        )

    def get_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        steps_per_epoch: int,
        warmup_ratio: float = 0.05,
        min_lr_ratio: float = 0.01,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        total_steps = max(1, num_epochs * steps_per_epoch)
        warmup_steps = int(total_steps * max(0.0, min(1.0, warmup_ratio)))

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=max(1, warmup_steps),
        )

        cosine_after = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps - warmup_steps),
            eta_min=optimizer.defaults["lr"] * min_lr_ratio,
        )

        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_after],
            milestones=[max(1, warmup_steps)],
        )

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None, prefix: str = "") -> None:
        if self.wandb_manager:
            self.wandb_manager.log_metrics(metrics, step=step, prefix=prefix)

    def save_checkpoint(self, epoch: int, metric_name: str, metric_value: float) -> None:
        ckpt_path = self.results_manager.save_checkpoint(
            run_id=self.experiment_name,
            model=self.model,
            epoch=epoch,
            metric_name=metric_name,
            metric_value=metric_value,
        )
        print(f"  -> Saved best model for {metric_name} at epoch {epoch} to {ckpt_path}")

    def _prepare_training(
        self,
        config: ExperimentConfig,
        device: torch.device,
        split: str,
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, int]:
        size_b = model_size_b(self.model)
        print(f"Training model with size: {size_b / MiB:.3f} MiB")

        self.model.to(device)
        self.training_num_steps = int(config.num_steps)
        optimizer = self.get_optimizer(
            float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )
        self.model.train()

        steps_per_epoch = self._get_num_batches(split)
        print(f"Steps per epoch: {steps_per_epoch}")

        scheduler = self.get_scheduler(
            optimizer,
            num_epochs=int(config.num_epochs),
            steps_per_epoch=steps_per_epoch,
            warmup_ratio=float(config.warmup_ratio),
            min_lr_ratio=float(config.min_lr_ratio),
        )
        total_steps = int(config.num_epochs) * steps_per_epoch
        warmup_steps = int(total_steps * float(config.warmup_ratio))
        min_lr = float(config.learning_rate) * float(config.min_lr_ratio)
        print(
            f"Using cosine LR schedule: warmup {warmup_steps} steps, "
            f"then cosine decay to {min_lr:.2e}"
        )

        return optimizer, scheduler, steps_per_epoch

    def _run_training_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        epoch: int,
        num_epochs: int,
        split: str,
        steps_per_epoch: int,
    ) -> tuple[float, dict[str, float]]:
        epoch_loss = 0.0
        epoch_component_sums: dict[str, float] = {}

        pbar = tqdm(
            range(steps_per_epoch),
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            leave=False,
        )

        for step in pbar:
            optimizer.zero_grad()
            loss_results = self.get_train_loss(split)
            loss = loss_results["loss"]
            loss_breakdown = {
                key: _to_float(value)
                for key, value in loss_results["log_values"].items()
            }
            if not loss_breakdown:
                loss_breakdown = {"total_loss": float(loss.detach().item())}

            for key, value in loss_breakdown.items():
                epoch_component_sums[key] = epoch_component_sums.get(key, 0.0) + value

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_threshold)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            epoch_loss += float(loss.detach().item())
            current_lr = optimizer.param_groups[0]["lr"]

            postfix = {
                "step": f"{step + 1}/{steps_per_epoch}",
                "loss": f"{loss.detach().item():.3f}",
                "lr": f"{current_lr:.2e}",
            }
            if "fm_loss_weighted" in loss_breakdown:
                postfix["fm"] = f"{loss_breakdown['fm_loss_weighted']:.3f}"
            elif "fm_loss" in loss_breakdown:
                postfix["fm"] = f"{loss_breakdown['fm_loss']:.3f}"
            if "psd_loss_weighted" in loss_breakdown:
                postfix["psd"] = f"{loss_breakdown['psd_loss_weighted']:.3f}"
            elif "psd_loss" in loss_breakdown:
                postfix["psd"] = f"{loss_breakdown['psd_loss']:.3f}"
            pbar.set_postfix(**postfix)

        avg_loss = epoch_loss / steps_per_epoch
        avg_components = {
            key: value / steps_per_epoch
            for key, value in epoch_component_sums.items()
        }
        avg_components.setdefault("total_loss", avg_loss)
        return avg_loss, avg_components

    def _run_validation(
        self,
        epoch: int,
        config: ExperimentConfig,
        device: torch.device,
        val_metrics_arr: list[dict[str, Any]],
    ) -> bool:
        validate_every = config.validate_every
        if not validate_every or (epoch + 1) % validate_every != 0:
            return False

        epoch_number = epoch + 1
        print(f"Running validation at epoch {epoch_number}...")
        request = self._build_split_eval_request(
            split="val",
            device=device,
            desc_prefix="Validating",
            kwargs={"num_steps": int(config.num_steps)},
        )
        result = self._evaluate_split(request)
        val_metrics = dict(result["metrics"])
        print(f"Validation metrics: {val_metrics}")

        val_metrics["epoch"] = epoch_number
        val_metrics_arr.append(val_metrics)
        self.log_metrics(val_metrics, step=epoch_number, prefix="val")
        improved = self._update_best_models(val_metrics, epoch_number)
        self._stalled_validations = 0 if improved else self._stalled_validations + 1
        stop = (
            self.early_stopping_patience > 0
            and self._stalled_validations >= self.early_stopping_patience
        )

        self.model.train()
        return stop

    def _log_training_epoch(
        self,
        epoch: int,
        num_epochs: int,
        avg_loss: float,
        avg_components: dict[str, float],
        current_lr: float,
    ) -> None:
        component_parts: list[str] = []
        for key in ("fm_loss_weighted", "psd_loss_weighted", "total_loss"):
            if key in avg_components:
                component_parts.append(f"{key}: {avg_components[key]:.4f}")
        component_suffix = f", {', '.join(component_parts)}" if component_parts else ""
        print(
            f"Epoch {epoch + 1}/{num_epochs} - avg loss: {avg_loss:.4f}, "
            f"lr: {current_lr:.2e}{component_suffix}"
        )

        train_metrics = {"loss": avg_loss, "epoch": epoch + 1, "lr": current_lr}
        train_metrics.update(avg_components)
        self.log_metrics(train_metrics, step=epoch + 1, prefix="train")

    def _finalize_training(
        self,
        loss_arr: list[float],
        val_metrics_arr: list[dict[str, Any]],
        config: ExperimentConfig,
    ) -> None:
        self.model.eval()
        self._print_best_models_summary()
        self._save_training_results(loss_arr, val_metrics_arr, config=config)

    def _update_best_models(self, val_metrics: dict[str, Any], epoch: int) -> bool:
        improved = False
        for metric_name, record in self._best_model_records.items():
            if metric_name not in val_metrics:
                continue

            current_value = float(val_metrics[metric_name])
            if current_value < record["value"]:
                print(
                    f"  -> New best {metric_name}: {current_value:.6f} "
                    f"(previous: {record['value']:.6f})"
                )
                record["value"] = current_value
                record["epoch"] = epoch
                self.save_checkpoint(epoch, metric_name, current_value)
                improved = True

        return improved

    def _print_best_models_summary(self) -> None:
        if not self._best_model_records:
            return

        print("\n" + "=" * 60)
        print("Best models summary:")
        for metric_name, record in self._best_model_records.items():
            if record["epoch"] is not None:
                print(f"  {metric_name}: {record['value']:.6f} at epoch {record['epoch']}")
        print("=" * 60 + "\n")

    def _save_training_results(
        self,
        loss_arr: list[float],
        val_metrics_arr: list[dict[str, Any]],
        config: ExperimentConfig | None = None,
    ) -> None:
        results = {
            "experiment_name": self.experiment_name,
            "loss_history": loss_arr,
            "validation_history": val_metrics_arr,
            "best_models": self.best_models,
            "stopped_epoch": self._stopped_epoch,
        }
        if config is not None:
            results["config"] = config.to_dict()

        results_path = self.results_manager.save_training_results(
            run_id=self.experiment_name,
            results=results,
        )
        print(f"Training results saved to {results_path}")


__all__ = ["TrainingLoopMixin"]
