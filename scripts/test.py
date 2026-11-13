"""Test evaluation with prediction caching.

Thin wrapper around ``os_flow_matching.runners.run_test_and_cache``.
Discovers trained runs, skips completed ones, and delegates evaluation
and prediction caching to the library.

Cache layout under ``results_{signal}/predictions/{run_id}/``::

    session_ranges.json                          # shared index mapping

    zeroshot/{checkpoint_metric}/                 # one global prediction set
        pred.pt, target.pt, source.pt

    fewshot/{checkpoint_metric}/{session_name}/   # per-session predictions
        pred.pt, target.pt, source.pt
"""

from __future__ import annotations

import json
from pathlib import Path

from os_flow_matching import (
    ExperimentConfig,
    ResultsManager,
    get_device,
    print_experiment_header,
    print_final_summary,
    run_test_and_cache,
    sampler_factory,
    zeroshot_dir,
    fewshot_dir,
    tensors_cached,
    session_ranges_cached,
    has_fewshot_checkpoints,
)

from settings import (
    CHECKPOINT_DIR,
    TEST_BEST_MODEL_METRICS,
)

REQUIRE_COMPLETED: bool = True
PREDICTIONS_DIR: Path = CHECKPOINT_DIR.parent / "predictions"


# ── Config discovery ──────────────────────────────────────────────────────────

def _load_experiment_configs(
    results_manager: ResultsManager,
    *,
    require_completed: bool,
) -> list[ExperimentConfig]:
    run_ids = sorted(m.name for m in results_manager.checkpoint_dir.iterdir() if m.is_dir())
    configs: list[ExperimentConfig] = []

    for run_id in run_ids:
        config, stored_run_id = results_manager.load_training_config(run_id=run_id)
        resolved_run_id = stored_run_id or config.make_run_id()
        if resolved_run_id != run_id:
            print(f"[SKIP] Run ID mismatch for {run_id}: config resolves to {resolved_run_id}")
            continue

        if require_completed and not results_manager.is_training_completed(run_id):
            print(f"[SKIP] Training not completed for {run_id}")
            continue

        configs.append(config)

    return configs


# ── Skip logic ────────────────────────────────────────────────────────────────

def _all_zeroshot_cached(run_id: str) -> bool:
    return session_ranges_cached(PREDICTIONS_DIR, run_id) and all(
        tensors_cached(zeroshot_dir(PREDICTIONS_DIR, run_id, m))
        for m in TEST_BEST_MODEL_METRICS
    )


def _fewshot_needs_work(run_id: str) -> bool:
    """Check if fewshot checkpoints exist that haven't been cached yet."""
    sr_path = PREDICTIONS_DIR / run_id / "session_ranges.json"
    if not sr_path.exists():
        return any(
            has_fewshot_checkpoints(CHECKPOINT_DIR, run_id, m)
            for m in TEST_BEST_MODEL_METRICS
        )

    ranges = json.loads(sr_path.read_text(encoding="utf-8"))
    for metric_name in TEST_BEST_MODEL_METRICS:
        if not has_fewshot_checkpoints(CHECKPOINT_DIR, run_id, metric_name):
            continue
        fewshot_ckpt_dir = CHECKPOINT_DIR / run_id / "fewshot"
        for sr in ranges:
            experiment = sr["experiment"]
            has_ckpt = any(fewshot_ckpt_dir.glob(f"fewshot_*_{metric_name}_{experiment}*"))
            if has_ckpt and not tensors_cached(fewshot_dir(PREDICTIONS_DIR, run_id, metric_name, experiment)):
                return True
    return False


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = get_device(0)
    test_output_dir = CHECKPOINT_DIR.parent / "test_results"

    results_manager = ResultsManager(
        checkpoint_dir=CHECKPOINT_DIR,
        test_output_dir=test_output_dir,
    )

    experiment_configs = _load_experiment_configs(
        results_manager,
        require_completed=REQUIRE_COMPLETED,
    )
    if not experiment_configs:
        raise RuntimeError("No valid checkpoints/configs found. Ensure models are trained.")

    print(f"{'=' * 60}")
    print("Test evaluation")
    print(f"  - Experiments found: {len(experiment_configs)}")
    print(f"  - Metrics: {TEST_BEST_MODEL_METRICS}")
    print(f"  - Predictions dir: {PREDICTIONS_DIR}")
    print(f"{'=' * 60}")

    skipped_count = 0

    for i, exp_config in enumerate(experiment_configs, 1):
        run_id = exp_config.make_run_id()

        # Skip if zeroshot is complete AND no uncached fewshot work remains.
        zs_done = (
            results_manager.is_test_completed(run_id, mode="zeroshot")
            and _all_zeroshot_cached(run_id)
        )
        fs_needed = _fewshot_needs_work(run_id)
        if zs_done and not fs_needed:
            print(f"\n[SKIP] {i}/{len(experiment_configs)}: {run_id} (zeroshot + fewshot complete)")
            skipped_count += 1
            continue

        if not results_manager.checkpoint_exists(run_id):
            print(f"\n[SKIP] {i}/{len(experiment_configs)}: {run_id} (checkpoint not found)")
            skipped_count += 1
            continue

        if REQUIRE_COMPLETED and not results_manager.is_training_completed(run_id):
            print(f"\n[SKIP] {i}/{len(experiment_configs)}: {run_id} (training not completed)")
            skipped_count += 1
            continue

        print_experiment_header(i, len(experiment_configs), run_id, f"Testing {exp_config.model_name}")

        with sampler_factory(**exp_config.get_sampler_kwargs()) as sampler:
            run_test_and_cache(
                config=exp_config,
                sampler=sampler,
                device=device,
                results_manager=results_manager,
                predictions_dir=PREDICTIONS_DIR,
            )

    print_final_summary(len(experiment_configs), skipped_count, results_manager.test_output_dir)
