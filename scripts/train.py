from __future__ import annotations

import warnings
from os_flow_matching import (
    get_device,
    run_training_experiment,
    WandbManager,
    ResultsManager,
    build_experiment_grid,
    print_sweep_summary,
    sampler_factory
)


from settings import (
    BATCH_SIZE,
    CONDITION_ON_X0,
    LEARNING_RATE,
    NUM_EPOCHS,
    NUM_STEPS,
    ROOT_DIR,
    DATASET_TYPE,
    SOURCE_SPEC,
    TARGET_SPEC,
    NUM_WORKERS,
    VAL_RATIO,
    TEST_RATIO,
    SEED,
    PATHS,
    USE_WANDB,
    VALIDATE_EVERY,
    EARLY_STOPPING_PATIENCE,
    WANDB_PROJECT,
    WINDOW_SIZE_SAMPLER,
    WINDOW_SIZE_MODEL,
    TRACKED_METRICS,
    CHECKPOINT_DIR,
    WARMUP_RATIO,
    MIN_LR_RATIO,
    WEIGHT_DECAY,
    DROPOUT,
    X0_NOISE_STD,
    PSD_RATIO_LOSS_WEIGHT,
    PSD_RATIO_BAND_HZ,
    PSD_STEP_SIZE,
    GRAD_CLIP_THRESHOLD,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

FORCE_TRAINING = False


if __name__ == "__main__":
    device = get_device()

    results_manager = ResultsManager(checkpoint_dir=CHECKPOINT_DIR)
    wandb_manager = WandbManager(
        WANDB_PROJECT,
        enabled=USE_WANDB
    ) if USE_WANDB else None

    experiment_configs = build_experiment_grid(
        paths=PATHS,
        seeds=SEED,
        batch_sizes=BATCH_SIZE,
        window_sizes_sampler=WINDOW_SIZE_SAMPLER,
        window_sizes_model=WINDOW_SIZE_MODEL,
        learning_rates=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        validate_every=VALIDATE_EVERY,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        root_dir=ROOT_DIR,
        dataset_type=DATASET_TYPE,
        source_spec=SOURCE_SPEC,
        target_spec=TARGET_SPEC,
        num_workers=NUM_WORKERS,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        condition_on_x0=CONDITION_ON_X0,
        dropout=DROPOUT,
        warmup_ratio=WARMUP_RATIO,
        min_lr_ratio=MIN_LR_RATIO,
        weight_decay=WEIGHT_DECAY,
        x0_noise_std=X0_NOISE_STD,
        psd_ratio_loss_weight=PSD_RATIO_LOSS_WEIGHT,
        psd_ratio_band_hz=PSD_RATIO_BAND_HZ,
        psd_step_size=PSD_STEP_SIZE,
        num_steps=NUM_STEPS,
        tracked_metrics=TRACKED_METRICS,
        grad_clip_threshold=GRAD_CLIP_THRESHOLD,
    )

    total_experiments = len(experiment_configs)
    print_sweep_summary(experiment_configs, action="train")

    for i, exp_config in enumerate(experiment_configs, 1):
        with sampler_factory(**exp_config.get_sampler_kwargs()) as sampler:
            sampler.to(device)
            run_training_experiment(
                config=exp_config,
                sampler=sampler,
                device=device,
                results_manager=results_manager,
                wandb_manager=wandb_manager,
                force_training=FORCE_TRAINING,
                experiment_index=i,
                total_experiments=total_experiments,
            )
