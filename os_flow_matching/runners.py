from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .utils.loggers import ExperimentConfig, ResultsManager, WandbManager
from .utils.general import crop_center
from .factories import trainer_factory
from settings import NUM_STEPS, TEST_BEST_MODEL_METRICS


# ── Printing helpers ──────────────────────────────────────────────────────────

def _print_metrics(metrics: dict[str, Any], title: str | None = None) -> None:
    def fmt(value: Any) -> str:
        return f"{value:.6f}" if isinstance(value, float) else str(value)

    if title:
        print(f"\n  {title}:")
    for key, value in sorted(metrics.items()):
        if isinstance(value, dict):
            continue
        if isinstance(value, float):
            print(f"    {key:35s} {fmt(value)}")


def _print_per_experiment(per_experiment: dict[str, dict[str, Any]]) -> None:
    def fmt(value: Any) -> str:
        return f"{value:.6f}" if isinstance(value, float) else str(value)

    for experiment, metrics in per_experiment.items():
        print(f"\n    [{experiment}]")
        for key, value in metrics.items():
            if key == "evaluation_history" and isinstance(value, list):
                print(f"      {key}: {len(value)} checkpoints")
                continue
            if isinstance(value, dict):
                continue
            print(f"      {key}: {fmt(value)}")


def _print_comparison(
    baseline: dict[str, Any],
    zeroshot: dict[str, Any],
    fewshot: dict[str, Any],
    tracked_metrics: tuple[str, ...],
) -> None:
    print(f"\n  {'-' * 56}")
    print("  Summary comparison (lower is better):")
    print(f"  {'-' * 56}")
    header = f"  {'metric':<25s} {'baseline':>10s} {'zero-shot':>10s} {'few-shot':>10s} {'delta':>10s}"
    print(header)
    print(f"  {'-' * 56}")
    for key in tracked_metrics:
        b_val = baseline.get(key)
        z_val = zeroshot.get(key)
        f_val = fewshot.get(key)
        if b_val is None or z_val is None or f_val is None:
            continue
        delta = z_val - f_val
        sign = "+" if delta >= 0 else ""
        print(f"  {key:<25s} {b_val:>10.4f} {z_val:>10.4f} {f_val:>10.4f} {sign}{delta:>9.4f}")
    print(f"  {'-' * 56}")
    b_n = baseline.get("num_samples", "?")
    z_n = zeroshot.get("num_samples", "?")
    f_n = fewshot.get("num_samples", "?")
    print(f"  {'num_samples':<25s} {str(int(b_n)):>10s} {str(int(z_n)):>10s} {str(int(f_n)):>10s}")


def print_final_summary(
    total: int,
    skipped: int,
    results_dir: Path = None,
) -> None:
    print(f"\n{'=' * 60}")
    print("All experiments processed!")
    print(f"  Total: {total}")
    print(f"  Skipped: {skipped}")
    print(f"  Processed: {total - skipped}")
    if results_dir:
        print(f"  Results: {results_dir}")
    print(f"{'=' * 60}")


# ── Prediction cache helpers ──────────────────────────────────────────────────

def zeroshot_dir(predictions_dir: Path, run_id: str, metric_name: str) -> Path:
    return predictions_dir / run_id / "zeroshot" / metric_name


def fewshot_dir(predictions_dir: Path, run_id: str, metric_name: str, session_name: str) -> Path:
    return predictions_dir / run_id / "fewshot" / metric_name / session_name


def tensors_cached(directory: Path) -> bool:
    return (directory / "pred.pt").exists() and (directory / "target.pt").exists() and (directory / "source.pt").exists()


def session_ranges_cached(predictions_dir: Path, run_id: str) -> bool:
    return (predictions_dir / run_id / "session_ranges.json").exists()


def has_fewshot_checkpoints(checkpoint_dir: Path, run_id: str, metric_name: str) -> bool:
    """Check if any fewshot checkpoints exist for this run."""
    fs_dir = checkpoint_dir / run_id / "fewshot"
    if not fs_dir.exists():
        return False
    return any(fs_dir.glob(f"fewshot_*_{metric_name}_*.pt"))


def _save_tensors(directory: Path, *, pred: torch.Tensor, target: torch.Tensor, source: torch.Tensor) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(pred.cpu(), directory / "pred.pt")
    torch.save(target.cpu(), directory / "target.pt")
    torch.save(source.cpu(), directory / "source.pt")


def _save_session_ranges(predictions_dir: Path, run_id: str, sampler) -> None:
    """Save test-split session ranges (subject_day → index mapping) once per run."""
    out_path = predictions_dir / run_id / "session_ranges.json"
    if out_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = sampler.datasets["test"]
    ranges = [
        {"experiment": str(item["experiment"]), "start": int(item["start"]), "end": int(item["end"])}
        for item in getattr(dataset, "_session_ranges", [])
    ]
    out_path.write_text(json.dumps(ranges, indent=2), encoding="utf-8")
    print(f"  Cached session ranges → {out_path} ({len(ranges)} sessions)")


@torch.no_grad()
def _cache_fewshot_predictions(
    trainer,
    config: ExperimentConfig,
    run_id: str,
    metric_name: str,
    x0_all: torch.Tensor,
    x1_all: torch.Tensor,
    cond_all: torch.Tensor | None,
    device: torch.device,
    sampler,
    predictions_dir: Path,
) -> None:
    dataset = sampler.datasets["test"]
    session_ranges = getattr(dataset, "_session_ranges", [])
    if not session_ranges:
        return

    for item in session_ranges:
        experiment = str(item["experiment"])
        start, end = int(item["start"]), int(item["end"])

        out_dir = fewshot_dir(predictions_dir, run_id, metric_name, experiment)
        if tensors_cached(out_dir):
            print(f"    [CACHED] fewshot {experiment}")
            continue

        fewshot_path = trainer.results_manager.get_fewshot_checkpoint_path(
            run_id, metric_name, experiment, checkpoint_type="best",
        )
        if not fewshot_path.exists():
            print(f"    [SKIP] No fewshot checkpoint for {experiment}")
            continue

        print(f"    Evaluating fewshot {experiment} (samples {start}:{end})...")
        try:
            history, _ = trainer.evaluate_data(
                x0_all=x0_all[start:end],
                x1_all=x1_all[start:end],
                cond_all=cond_all[start:end] if cond_all is not None else None,
                device=device,
                num_steps=NUM_STEPS,
                metric_name=metric_name,
                metric_names=(metric_name,),
                model_window_size=config.window_size_model,
                fewshot_experiment=experiment,
                fewshot_checkpoint_type="best",
            )
        except Exception as e:
            print(f"    Error evaluating fewshot {experiment}: {e}")
            continue

        pred = history["pred_traj"][:, -1]
        target = history["target_final"]
        crop_size = target.shape[-1]
        x0_session = trainer.transform.backward(x0_all[start:end])
        source = crop_center(x0_session, crop_size) if x0_session.shape[-1] != crop_size else x0_session

        _save_tensors(out_dir, pred=pred, target=target, source=source)
        print(f"    Cached fewshot predictions → {out_dir}")


# ── Training ──────────────────────────────────────────────────────────────────

def run_training_experiment(
    config: ExperimentConfig,
    sampler,
    device: torch.device,
    results_manager: ResultsManager,
    wandb_manager: WandbManager | None = None,
    *,
    force_training: bool = False,
    experiment_index: int | None = None,
    total_experiments: int | None = None,
) -> bool:
    run_id = config.make_run_id()

    if results_manager.is_training_completed(run_id) and not force_training:
        if experiment_index is not None and total_experiments is not None:
            print(f"\n[SKIP] Experiment {experiment_index}/{total_experiments}: {run_id} (already completed)")
        else:
            print(f"\n[SKIP] {run_id} (already completed)")
        return False

    results_manager.save_training_config(run_id=run_id, config=config)
    if experiment_index is not None and total_experiments is not None:
        from os_flow_matching.utils import print_experiment_header
        print_experiment_header(
            experiment_index,
            total_experiments,
            run_id,
            f"Training {config.model_name}",
        )

    if wandb_manager:
        wandb_manager.start_run(
            config,
            config.model_name,
            config.model_kwargs,
        )

    trainer = trainer_factory(config, sampler, device, results_manager, wandb_manager)

    print("Computing baseline metrics (source vs target)...")
    baseline_metrics = trainer.compute_baseline_metrics(device=device, split="val")
    print(f"Baseline metrics: {baseline_metrics}")
    trainer.log_metrics(baseline_metrics, step=0, prefix="val")

    loss_history, _ = trainer.train(config=config, device=device)

    final_loss = loss_history[-1] if loss_history else float("nan")
    print(f"{config.model_name} training completed. Final loss: {final_loss}")

    if wandb_manager:
        wandb_manager.finish_run(final_loss, trainer.best_models)

    return True


# ── Few-shot ──────────────────────────────────────────────────────────────────

def run_fewshot_experiment(
    config: ExperimentConfig,
    sampler,
    device: torch.device,
    results_manager: ResultsManager,
    *,
    checkpoint_metric: str,
    num_steps: int,
    support_count: int,
    finetune_epochs: int,
    finetune_lr: float,
    weight_decay: float,
    finetune_batch_size: int,
    test_every_k_epochs: int,
    force_training: bool = False,
    force_evaluate: bool = False,
) -> bool:
    run_id = config.make_run_id()

    if not results_manager.is_training_completed(run_id):
        print(f"\n[SKIP] {run_id} (training not completed)")
        return False

    output_dir = results_manager.test_output_dir
    if output_dir is None:
        raise ValueError("results_manager.test_output_dir is not configured for few-shot runs")
    output_file = output_dir / f"{run_id}_fewshot_results.json"
    output_exists = output_file.exists()

    if output_exists and not force_training and not force_evaluate:
        print(f"\n[INFO] {run_id} (few-shot output already exists; re-running evaluation)")

    trainer = trainer_factory(config, sampler, device, results_manager)

    try:
        checkpoint = trainer.load_checkpoint(checkpoint_metric, device=device)
        print(
            f"  Loaded checkpoint: best_{checkpoint_metric} "
            f"(epoch {checkpoint['epoch']}, value {checkpoint['metric_value']:.6f})"
        )
    except FileNotFoundError:
        print(f"  [WARN] Checkpoint for {checkpoint_metric} not found; skipping.")
        return False

    print("\n  [1/3] Baseline - raw source vs target distance (no model)")
    baseline_metrics = trainer.compute_baseline_metrics(device=device, split="test")
    _print_metrics(baseline_metrics, "Baseline")

    print("\n  [2/3] Zero-shot - trained checkpoint without few-shot fine-tuning")
    zeroshot_results = trainer.transfer_learn_and_evaluate(
        device=device,
        support_count=0,
        split="test",
        finetune_epochs=0,
        finetune_lr=finetune_lr,
        weight_decay=weight_decay,
        num_steps=num_steps,
    )
    _print_metrics(zeroshot_results["aggregate"], "Zero-shot aggregate")

    print(
        f"\n  [3/3] Few-shot - fine-tune on {support_count} support samples/session "
        "before test evaluation"
    )
    print(
        f"        finetune_epochs={finetune_epochs}, lr={finetune_lr}, "
        f"batch_size={finetune_batch_size}, test_every_k_epochs={test_every_k_epochs}"
    )
    train_fewshot = force_training or not output_exists
    save_fewshot_models = train_fewshot
    if not train_fewshot:
        print("        Re-eval mode: preserving existing few-shot checkpoints")
    fewshot_results = trainer.transfer_learn_and_evaluate(
        device=device,
        support_count=support_count,
        split="test",
        finetune_epochs=finetune_epochs,
        finetune_lr=finetune_lr,
        weight_decay=weight_decay,
        finetune_batch_size=finetune_batch_size,
        num_steps=num_steps,
        train_fewshot=train_fewshot,
        fewshot_checkpoint_metric=checkpoint_metric,
        save_fewshot_models=save_fewshot_models,
        test_every_k_epochs=test_every_k_epochs,
    )

    _print_metrics(fewshot_results["aggregate"], "Few-shot aggregate")

    _print_comparison(
        baseline=baseline_metrics,
        zeroshot=zeroshot_results["aggregate"],
        fewshot=fewshot_results["aggregate"],
        tracked_metrics=trainer.tracked_metrics,
    )
    _print_per_experiment(fewshot_results["per_experiment"])

    experiment_results = {
        "run_id": run_id,
        "model_name": config.model_name,
        "hyperparameters": config.to_dict(),
        "support_count": support_count,
        "finetune_epochs": finetune_epochs,
        "finetune_lr": finetune_lr,
        "weight_decay": weight_decay,
        "finetune_batch_size": finetune_batch_size,
        "test_every_k_epochs": test_every_k_epochs,
        "num_steps": num_steps,
        "checkpoint_metric": checkpoint_metric,
        "baseline_metrics": baseline_metrics,
        "zeroshot": zeroshot_results,
        "fewshot": fewshot_results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(experiment_results, f, indent=2, default=str)
    print(f"\n  Saved few-shot results to: {output_file}")

    return True


# ── Test + prediction caching ─────────────────────────────────────────────────

@torch.no_grad()
def run_test_and_cache(
    config: ExperimentConfig,
    sampler,
    device: torch.device,
    results_manager: ResultsManager,
    predictions_dir: Path,
) -> None:
    run_id = config.make_run_id()

    sampler.to(device)
    trainer = trainer_factory(config, sampler, device, results_manager)

    # Pre-load all test data once.
    x0_all, x1_all, cond_all = sampler.get_test_data(device=torch.device("cpu"))

    # Save session ranges (subject_day → index mapping).
    _save_session_ranges(predictions_dir, run_id, sampler)

    # Baseline metrics (source vs target, no model).
    print("\n  Computing baseline metrics...")
    baseline_metrics = trainer.compute_baseline_metrics(device=device, split="test")
    _print_metrics(baseline_metrics, "Baseline")

    experiment_results = {
        "run_id": run_id,
        "model_name": config.model_name,
        "hyperparameters": config.to_dict(),
        "baseline_metrics": baseline_metrics,
        "test_results": {},
        "num_steps": NUM_STEPS,
        "test_best_model_metrics": list(TEST_BEST_MODEL_METRICS),
    }

    for metric_name in TEST_BEST_MODEL_METRICS:
        print(f"\n  Evaluating checkpoint: {metric_name}")

        # ── Zeroshot ──────────────────────────────────────────────
        zs_dir = zeroshot_dir(predictions_dir, run_id, metric_name)
        if tensors_cached(zs_dir):
            print(f"  [CACHED] zeroshot {metric_name}")
        else:
            try:
                history, metrics = trainer.evaluate_data(
                    x0_all=x0_all,
                    x1_all=x1_all,
                    cond_all=cond_all,
                    device=device,
                    num_steps=NUM_STEPS,
                    metric_name=metric_name,
                    metric_names=tuple(TEST_BEST_MODEL_METRICS),
                    model_window_size=config.window_size_model,
                )
            except FileNotFoundError as e:
                print(f"  Checkpoint not found for {metric_name}: {e}")
                continue
            except Exception as e:
                print(f"  Error evaluating {metric_name}: {e}")
                continue

            metrics["checkpoint_metric"] = metric_name
            experiment_results["test_results"][metric_name] = metrics
            _print_metrics(metrics, f"Results ({metric_name})")

            pred = history["pred_traj"][:, -1]
            target = history["target_final"]
            crop_size = target.shape[-1]
            x0_ts = trainer.transform.backward(x0_all)
            source = crop_center(x0_ts, crop_size) if x0_ts.shape[-1] != crop_size else x0_ts
            _save_tensors(zs_dir, pred=pred, target=target, source=source)
            print(f"  Cached zeroshot predictions → {zs_dir}")

        # ── Fewshot ───────────────────────────────────────────────
        if has_fewshot_checkpoints(results_manager.checkpoint_dir, run_id, metric_name):
            print(f"\n  Caching fewshot predictions for {metric_name}:")
            _cache_fewshot_predictions(
                trainer, config, run_id, metric_name,
                x0_all, x1_all, cond_all, device, sampler,
                predictions_dir,
            )

    # Save zeroshot test results JSON.
    if experiment_results["test_results"]:
        output_file = results_manager.save_test_results(run_id, experiment_results, mode="zeroshot")
        print(f"\n  Saved zeroshot test results → {output_file}")

    # Cleanup.
    trainer.model.cpu()
    if hasattr(trainer, "path"):
        trainer.path.cpu()
    del trainer
    torch.cuda.empty_cache()
