from __future__ import annotations

import inspect
import warnings

from .dataset.sampleable import BeatDatasetSampler, TSDatasetSampler
from .utils.loggers import ExperimentConfig, ResultsManager, WandbManager
from .trainer.base import Trainer
from .trainer.supervised import SupervisedTrainer
from .trainer.flow_matching import VanillaFMTrainer


def _configure_sampler(sampler_cls, sampler_kwargs):
    sampler_params = set(inspect.signature(sampler_cls.__init__).parameters) - {"self"}
    ignored = sorted(set(sampler_kwargs) - sampler_params)
    if ignored:
        warnings.warn(
            f"{sampler_cls.__name__} ignores settings: {', '.join(ignored)}",
            RuntimeWarning,
            stacklevel=3,
        )
    filtered_kwargs = {
        key: value for key, value in sampler_kwargs.items() if key in sampler_params
    }
    return sampler_cls(**filtered_kwargs)


def sampler_factory(
    *,
    root_dir: str,
    source_spec,
    target_spec,
    split: str = "train",
    window_size: int = 2048,
    stride: int | None = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 10,
    fs: int = 500,
    batch_size: int = 256,
    num_workers: int = 4,
    prefetch_factor: int | None = 4,
    pin_memory: bool = True,
    dataset_type: str | None = None,
    transform=None,
):
    if dataset_type is None:
        raise ValueError("dataset_type must be specified explicitly to avoid ambiguity.")

    sampler_kwargs = {
        "root_dir": root_dir,
        "source_spec": source_spec,
        "target_spec": target_spec,
        "split": split,
        "window_size": window_size,
        "stride": stride,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
        "fs": fs,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        "pin_memory": pin_memory,
        "transform": transform,
    }
    sampler_cls = BeatDatasetSampler if dataset_type == "beat" else TSDatasetSampler
    return _configure_sampler(sampler_cls, sampler_kwargs)


def trainer_factory(
    config: ExperimentConfig,
    sampler,
    device,
    results_manager: ResultsManager,
    wandb_manager: WandbManager | None = None,
) -> Trainer:
    run_id = config.make_run_id()
    model = config.model_class(**config.model_kwargs)

    if config.is_supervised:
        return SupervisedTrainer(
            sampler=sampler,
            model=model,
            tracked_metrics=config.tracked_metrics,
            early_stopping_patience=config.early_stopping_patience,
            results_manager=results_manager,
            experiment_name=run_id,
            wandb_manager=wandb_manager,
            condition_on_x0=config.condition_on_x0,
            x0_noise_std=config.x0_noise_std,
            grad_clip_threshold=config.grad_clip_threshold,
        )
    else:
        path = config.path_class(sampler=sampler).to(device)

        return VanillaFMTrainer(
            path=path,
            model=model,
            tracked_metrics=config.tracked_metrics,
            early_stopping_patience=config.early_stopping_patience,
            results_manager=results_manager,
            experiment_name=run_id,
            wandb_manager=wandb_manager,
            condition_on_x0=config.condition_on_x0,
            x0_noise_std=config.x0_noise_std,
            grad_clip_threshold=config.grad_clip_threshold,
            channel_weights=config.channel_weights,
            psd_ratio_loss_weight=config.psd_ratio_loss_weight,
            psd_ratio_band_hz=config.psd_ratio_band_hz,
            training_num_steps=config.num_steps,
            psd_step_size=config.psd_step_size,
        )
