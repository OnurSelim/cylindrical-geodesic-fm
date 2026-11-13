from __future__ import annotations

from typing import Any
import torch
from torch import Tensor


class TransferEvaluationMixin:
    def transfer_learn_and_evaluate(
        self,
        device: torch.device,
        support_count: int,
        split: str = "test",
        finetune_epochs: int = 1,
        finetune_lr: float = 1e-5,
        weight_decay: float = 1e-4,
        finetune_batch_size: int | None = None,
        save_fewshot_models: bool = False,
        test_every_k_epochs: int | None = None,
        train_fewshot: bool = True,
        fewshot_checkpoint_metric: str | None = None,
        num_steps: int = 40,
    ) -> dict[str, Any]:
        should_train_fewshot = support_count > 0 and train_fewshot
        should_save_fewshot_models = support_count > 0 and should_train_fewshot and save_fewshot_models

        if test_every_k_epochs is not None and test_every_k_epochs < 0:
            raise ValueError("test_every_k_epochs must be >= 0 or None")
        if test_every_k_epochs == 0:
            test_every_k_epochs = None

        sampler, datasets = self._get_sampler_and_datasets()
        session_ranges = list(getattr(datasets[split], "_session_ranges", []))
        sessions = self._build_session_indices(session_ranges, support_count)
        base_state = self._snapshot_model_state()
        support_batch_size = sampler.batch_size if finetune_batch_size is None else int(finetune_batch_size)
        if support_batch_size <= 0:
            raise ValueError("finetune_batch_size must be > 0")

        per_experiment: dict[str, dict[str, Any]] = {}
        for session in sessions:
            if not session["query_indices"]:
                continue

            self._restore_model_state(base_state)
            self.model.to(device)
            use_saved_fewshot_checkpoint = support_count > 0 and not should_train_fewshot
            if use_saved_fewshot_checkpoint:
                learned_checkpoints = self._load_saved_transfer_checkpoint(
                    session=session,
                    device=device,
                    metric_name=fewshot_checkpoint_metric,
                )
                if not learned_checkpoints:
                    continue
            else:
                effective_finetune_epochs = finetune_epochs if session["support_indices"] else 0
                eval_epochs = self._resolve_transfer_eval_epochs(
                    finetune_epochs=effective_finetune_epochs,
                    test_every_k_epochs=test_every_k_epochs,
                )
                learned_checkpoints = self._transfer_learn(
                    session=session,
                    split=split,
                    support_batch_size=support_batch_size,
                    finetune_epochs=effective_finetune_epochs,
                    eval_epochs=eval_epochs,
                    finetune_lr=finetune_lr,
                    weight_decay=weight_decay,
                )
            stage_result = self._transfer_evaluate(
                session=session,
                split=split,
                device=device,
                num_steps=num_steps,
                learned_checkpoints=learned_checkpoints,
                save_fewshot_models=should_save_fewshot_models,
            )
            evaluation_history = stage_result["evaluation_history"]
            best_states_by_metric = stage_result["best_states_by_metric"]
            last_state_dict = stage_result["last_state_dict"]
            last_metrics_by_metric = stage_result["last_metrics_by_metric"]

            per_experiment[session["experiment"]] = self._serialize_transfer_session_result(
                session=session,
                evaluation_history=evaluation_history,
            )

            if should_save_fewshot_models:
                for metric_name, best_info in best_states_by_metric.items():
                    self.results_manager.save_fewshot_checkpoint(
                        run_id=self.experiment_name,
                        state_dict=best_info["state_dict"],
                        epoch=int(best_info["epoch"]),
                        metric_name=metric_name,
                        metric_value=float(best_info["metric_value"]),
                        experiment=session["experiment"],
                        checkpoint_type="best",
                    )
                if last_state_dict is not None and evaluation_history:
                    last_epoch = int(evaluation_history[-1].get("finetune_epoch", 0))
                    for metric_name, metric_value in last_metrics_by_metric.items():
                        self.results_manager.save_fewshot_checkpoint(
                            run_id=self.experiment_name,
                            state_dict=last_state_dict,
                            epoch=last_epoch,
                            metric_name=metric_name,
                            metric_value=float(metric_value),
                            experiment=session["experiment"],
                            checkpoint_type="last",
                        )
                print(f"    Saved best and last fine-tuned models for session {session['experiment']}")

        self._restore_model_state(base_state)
        self.model.eval()

        if not per_experiment:
            return {"per_experiment": {}, "aggregate": {}}

        aggregate = self._aggregate_transfer_metrics(per_experiment)
        return {"per_experiment": per_experiment, "aggregate": aggregate}

    def _load_saved_transfer_checkpoint(
        self,
        session: dict[str, Any],
        device: torch.device,
        metric_name: str | None,
    ) -> list[dict[str, Any]]:
        resolved_metric = metric_name or (self.tracked_metrics[0] if self.tracked_metrics else None)
        if not resolved_metric:
            print(f"    [SKIP] {session['experiment']}: no metric configured to load few-shot checkpoint")
            return []

        try:
            checkpoint = self.results_manager.load_fewshot_checkpoint(
                run_id=self.experiment_name,
                model=self.model,
                metric_name=resolved_metric,
                experiment=session["experiment"],
                device=device,
                checkpoint_type="best",
            )
        except FileNotFoundError:
            print(
                f"    [SKIP] {session['experiment']}: missing saved few-shot checkpoint "
                f"(metric={resolved_metric})"
            )
            return []

        print(
            f"    [{session['experiment']}] loaded saved few-shot checkpoint: "
            f"epoch {int(checkpoint['epoch'])}, {resolved_metric}={float(checkpoint['metric_value']):.6f}"
        )
        return [
            {
                "epoch": int(checkpoint.get("epoch", 0)),
                "state_dict": self._snapshot_model_state(),
            }
        ]

    def _transfer_learn(
        self,
        session: dict[str, Any],
        split: str,
        support_batch_size: int,
        finetune_epochs: int,
        eval_epochs: list[int],
        finetune_lr: float,
        weight_decay: float,
    ) -> list[dict[str, Any]]:
        self.model.train()
        optimizer: torch.optim.Optimizer | None = None
        trained_epochs = 0
        checkpoints: list[dict[str, Any]] = []

        for eval_epoch in eval_epochs:
            optimizer, chunk_losses = self._finetune_support_set(
                session=session,
                split=split,
                batch_size=support_batch_size,
                finetune_epochs=eval_epoch - trained_epochs,
                finetune_lr=finetune_lr,
                weight_decay=weight_decay,
                optimizer=optimizer,
            )

            if chunk_losses:
                for offset, avg_loss in enumerate(chunk_losses, start=1):
                    local_epoch = trained_epochs + offset
                    print(
                        f"    [{session['experiment']}] fine-tune epoch "
                        f"{local_epoch}/{finetune_epochs} - avg train loss: {avg_loss:.6f}"
                    )
            trained_epochs = eval_epoch
            checkpoints.append(
                {
                    "epoch": eval_epoch,
                    "state_dict": self._snapshot_model_state(),
                }
            )

        return checkpoints

    def _transfer_evaluate(
        self,
        session: dict[str, Any],
        split: str,
        device: torch.device,
        num_steps: int,
        learned_checkpoints: list[dict[str, Any]],
        save_fewshot_models: bool,
    ) -> dict[str, Any]:
        evaluation_history: list[dict[str, Any]] = []
        best_states_by_metric: dict[str, dict[str, Any]] = {}
        last_state_dict: dict[str, Any] | None = None
        last_metrics_by_metric: dict[str, float] = {}

        for checkpoint in learned_checkpoints:
            eval_epoch = int(checkpoint["epoch"])
            self._restore_model_state(checkpoint["state_dict"])
            self.model.to(device)
            self.model.eval()

            metrics = self._evaluate_transfer_query(
                session=session,
                split=split,
                device=device,
                num_steps=num_steps,
            )
            metrics["finetune_epoch"] = eval_epoch
            evaluation_history.append(metrics)

            metric_parts = [
                f"{metric_name}={float(metrics[metric_name]):.6f}"
                for metric_name in self.tracked_metrics
                if metric_name in metrics
            ]
            metrics_line = ", ".join(metric_parts) if metric_parts else "no tracked metrics"
            print(f"    [{session['experiment']}] eval @ epoch {eval_epoch}: {metrics_line}")

            if save_fewshot_models:
                self._update_best_fewshot_states(
                    best_states_by_metric=best_states_by_metric,
                    metrics=metrics,
                    eval_epoch=eval_epoch,
                )
                last_state_dict = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in self.model.state_dict().items()
                }
                last_metrics_by_metric = {
                    metric_name: float(metrics[metric_name])
                    for metric_name in self.tracked_metrics
                    if metric_name in metrics
                }

        return {
            "evaluation_history": evaluation_history,
            "best_states_by_metric": best_states_by_metric,
            "last_state_dict": last_state_dict,
            "last_metrics_by_metric": last_metrics_by_metric,
        }

    @staticmethod
    def _build_session_indices(
        session_ranges: list[dict[str, Any]],
        support_count: int,
    ) -> list[dict[str, Any]]:
        if support_count < 0:
            raise ValueError("support_count must be >= 0")

        sessions: list[dict[str, Any]] = []
        for item in session_ranges:
            experiment = str(item["experiment"])
            start = int(item["start"])
            end = int(item["end"])
            split_idx = min(start + support_count, end)
            sessions.append(
                {
                    "experiment": experiment,
                    "support_indices": list(range(start, split_idx)),
                    "query_indices": list(range(split_idx, end)),
                }
            )
        return sessions

    @staticmethod
    def _iter_index_batches(
        indices: list[int],
        batch_size: int,
        shuffle: bool,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not indices:
            return

        if shuffle:
            perm = torch.randperm(len(indices)).tolist()
            ordered = [indices[i] for i in perm]
        else:
            ordered = list(indices)

        for i in range(0, len(ordered), batch_size):
            yield ordered[i:i + batch_size]

    def _snapshot_model_state(self) -> dict[str, Tensor]:
        return {
            name: tensor.detach().clone()
            for name, tensor in self.model.state_dict().items()
        }

    def _restore_model_state(self, state_dict: dict[str, Tensor]) -> None:
        self.model.load_state_dict(state_dict)

    def _finetune_support_set(
        self,
        session: dict[str, Any],
        split: str,
        batch_size: int,
        finetune_epochs: int,
        finetune_lr: float,
        weight_decay: float,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> tuple[torch.optim.Optimizer | None, list[float]]:
        if not session["support_indices"] or finetune_epochs <= 0:
            return optimizer, []

        self.model.train()
        if optimizer is None:
            optimizer = self.get_optimizer(lr=finetune_lr, weight_decay=weight_decay)

        epoch_losses: list[float] = []
        for _ in range(finetune_epochs):
            epoch_loss_sum = 0.0
            epoch_steps = 0
            for batch_indices in self._iter_index_batches(
                session["support_indices"],
                batch_size,
                shuffle=True,
            ):
                optimizer.zero_grad()
                step_result = self.get_train_loss_at(batch_indices, split)
                loss = step_result["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_threshold)
                optimizer.step()
                epoch_loss_sum += float(loss.detach().item())
                epoch_steps += 1
            epoch_losses.append(epoch_loss_sum / max(1, epoch_steps))
        return optimizer, epoch_losses

    @staticmethod
    def _resolve_transfer_eval_epochs(
        finetune_epochs: int,
        test_every_k_epochs: int | None,
    ) -> list[int]:
        if finetune_epochs <= 0:
            return [0]
        if test_every_k_epochs is None or test_every_k_epochs <= 0:
            return [finetune_epochs]

        eval_epochs = list(range(test_every_k_epochs, finetune_epochs + 1, test_every_k_epochs))
        if not eval_epochs or eval_epochs[-1] != finetune_epochs:
            eval_epochs.append(finetune_epochs)
        return eval_epochs

    def _evaluate_transfer_query(
        self,
        session: dict[str, Any],
        split: str,
        device: torch.device,
        num_steps: int,
    ) -> dict[str, Any]:
        request = self._build_split_eval_request(
            split=split,
            device=device,
            desc_prefix=f"Evaluating {session['experiment']}",
            kwargs={"num_steps": num_steps},
        )
        result = self._evaluate_indices(session["query_indices"], request)
        return dict(result["metrics"])

    def _update_best_fewshot_states(
        self,
        best_states_by_metric: dict[str, dict[str, Any]],
        metrics: dict[str, Any],
        eval_epoch: int,
    ) -> None:
        for metric_name in self.tracked_metrics:
            if metric_name not in metrics:
                continue

            current_value = float(metrics[metric_name])
            best_info = best_states_by_metric.get(metric_name)
            if best_info is not None and current_value >= float(best_info["metric_value"]):
                continue

            best_states_by_metric[metric_name] = {
                "metric_value": current_value,
                "epoch": eval_epoch,
                "state_dict": {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in self.model.state_dict().items()
                },
            }

    def _best_epochs_by_metric(
        self,
        evaluation_history: list[dict[str, Any]],
    ) -> dict[str, dict[str, float | int]]:
        best: dict[str, dict[str, float | int]] = {}
        for metric_name in self.tracked_metrics:
            candidates = [entry for entry in evaluation_history if metric_name in entry]
            if not candidates:
                continue
            best_entry = min(candidates, key=lambda entry: float(entry[metric_name]))
            best[metric_name] = {
                "epoch": int(best_entry["finetune_epoch"]),
                "value": float(best_entry[metric_name]),
            }
        return best

    def _serialize_transfer_session_result(
        self,
        session: dict[str, Any],
        evaluation_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        final_metrics = dict(evaluation_history[-1])
        best_epoch_by_metric = self._best_epochs_by_metric(evaluation_history)

        selected_metrics = {
            key: value
            for key, value in final_metrics.items()
            if key not in self.tracked_metrics and key != "finetune_epoch"
        }
        for metric_name in self.tracked_metrics:
            best_info = best_epoch_by_metric.get(metric_name)
            if best_info is not None:
                selected_metrics[metric_name] = float(best_info["value"])
            elif metric_name in final_metrics:
                selected_metrics[metric_name] = final_metrics[metric_name]

        selected_metrics.update(
            {
                "experiment": session["experiment"],
                "support_samples": len(session["support_indices"]),
                "query_samples": len(session["query_indices"]),
                "final_epoch": int(final_metrics.get("finetune_epoch", 0)),
                "tested_epochs": [
                    entry["finetune_epoch"]
                    for entry in evaluation_history
                ],
                "evaluation_history": evaluation_history,
                "best_epoch_by_metric": best_epoch_by_metric,
            }
        )
        return selected_metrics

    def _aggregate_transfer_metrics(
        self,
        per_experiment: dict[str, dict[str, Any]],
    ) -> dict[str, float]:
        if not per_experiment:
            return {}

        aggregate: dict[str, float] = {}
        metric_keys = [
            key
            for key in self.tracked_metrics
            if key in next(iter(per_experiment.values()))
        ]
        total_query = sum(value["query_samples"] for value in per_experiment.values())

        for key in metric_keys:
            weighted_sum = sum(value[key] * value["query_samples"] for value in per_experiment.values())
            aggregate[key] = weighted_sum / max(1, total_query)

        aggregate["num_experiments"] = float(len(per_experiment))
        aggregate["num_samples"] = float(total_query)
        return aggregate


__all__ = ["TransferEvaluationMixin"]
