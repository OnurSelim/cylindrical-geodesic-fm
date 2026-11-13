from __future__ import annotations

from os_flow_matching import (
    ExperimentConfig,
    ResultsManager,
    get_device,
    print_final_summary,
    run_fewshot_experiment,
    sampler_factory,
)

from settings import CHECKPOINT_DIR

# ── Configuration ─────────────────────────────────────────────────────────────
CHECKPOINT_METRIC: str = "ht_intrinsic_mean"
NUM_STEPS: int = 100
SEEDS: list[int] | None = [6, 10, 11, 13]  # None = all
REQUIRE_COMPLETED: bool = True

FEWSHOT_SUPPORT_COUNT: int = 512
FEWSHOT_FINETUNE_EPOCHS: int = 50
FEWSHOT_FINETUNE_LR: float = 1e-4
FEWSHOT_WEIGHT_DECAY: float = 1e-2
FEWSHOT_FINETUNE_BATCH_SIZE: int = 128
FEWSHOT_TEST_EVERY_K_EPOCHS: int = 0  # 0 = evaluate the final checkpoint only (paper)
FORCE_TRAINING: bool = False
FORCE_EVALUATE: bool = True


# ── Config discovery ──────────────────────────────────────────────────────────

def _load_experiment_configs(
    results_manager: ResultsManager,
) -> list[ExperimentConfig]:
    run_ids = sorted(m.name for m in results_manager.checkpoint_dir.iterdir() if m.is_dir())
    configs: list[ExperimentConfig] = []

    for run_id in run_ids:
        try:
            config, stored_run_id = results_manager.load_training_config(run_id=run_id)
        except Exception as e:
            print(f"[SKIP] Could not load config for {run_id}: {e}")
            continue

        resolved_run_id = stored_run_id or config.make_run_id()
        if resolved_run_id != run_id:
            print(f"[SKIP] Run ID mismatch for {run_id}: config resolves to {resolved_run_id}")
            continue

        if REQUIRE_COMPLETED and not results_manager.is_training_completed(run_id):
            print(f"[SKIP] Training not completed for {run_id}")
            continue

        if SEEDS is not None and config.seed not in SEEDS:
            continue

        configs.append(config)

    return configs


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    device = get_device(0)

    results_manager = ResultsManager(
        checkpoint_dir=CHECKPOINT_DIR,
        test_output_dir=CHECKPOINT_DIR.parent / "test_results",
    )

    experiment_configs = _load_experiment_configs(results_manager)
    if not experiment_configs:
        raise RuntimeError("No valid experiment configs found. Ensure models are trained.")

    print(f"{'=' * 60}")
    print("Few-shot fine-tuning")
    print(f"  - Experiments found  : {len(experiment_configs)}")
    print(f"  - Checkpoint metric  : {CHECKPOINT_METRIC}")
    print(f"  - Seeds              : {SEEDS or 'all'}")
    print(f"  - Support count      : {FEWSHOT_SUPPORT_COUNT}")
    print(f"  - Fine-tune epochs   : {FEWSHOT_FINETUNE_EPOCHS}")
    print(f"  - Fine-tune LR       : {FEWSHOT_FINETUNE_LR}")
    print(f"  - Fine-tune batch    : {FEWSHOT_FINETUNE_BATCH_SIZE}")
    print(f"  - Test every k epochs: {FEWSHOT_TEST_EVERY_K_EPOCHS}")
    print(f"  - Num steps          : {NUM_STEPS}")
    print(f"{'=' * 60}")

    skipped_count = 0

    for i, config in enumerate(experiment_configs, 1):
        run_id = config.make_run_id()

        print(f"\n{'─' * 60}")
        print(f"[{i}/{len(experiment_configs)}] {config.model_name} — {run_id}")
        print(f"{'─' * 60}")

        with sampler_factory(**config.get_sampler_kwargs()) as sampler:
            sampler.to(device)
            completed = run_fewshot_experiment(
                config=config,
                sampler=sampler,
                device=device,
                results_manager=results_manager,
                checkpoint_metric=CHECKPOINT_METRIC,
                num_steps=NUM_STEPS,
                support_count=FEWSHOT_SUPPORT_COUNT,
                finetune_epochs=FEWSHOT_FINETUNE_EPOCHS,
                finetune_lr=FEWSHOT_FINETUNE_LR,
                weight_decay=FEWSHOT_WEIGHT_DECAY,
                finetune_batch_size=FEWSHOT_FINETUNE_BATCH_SIZE,
                test_every_k_epochs=FEWSHOT_TEST_EVERY_K_EPOCHS,
                force_training=FORCE_TRAINING,
                force_evaluate=FORCE_EVALUATE,
            )
            if not completed:
                skipped_count += 1

    print_final_summary(len(experiment_configs), skipped_count, results_manager.test_output_dir)


if __name__ == "__main__":
    main()
