from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from itertools import product
from pathlib import Path
from typing import Dict, Any, Union, Sequence, List, Tuple

import torch

from ..model import infer_model_kwargs_from_specs


# region Wandb Integration
# ---------------------------------------------------------------------------
# Wandb Integration
# ---------------------------------------------------------------------------

class WandbManager:
    def __init__(self, project: str, enabled: bool = True):
        self.project = project
        self.enabled = enabled
        self._wandb = None

    def _get_wandb(self):
        if self._wandb is None:
            import wandb
            self._wandb = wandb
        return self._wandb

    def start_run(
        self,
        config: ExperimentConfig,
        model_name: str,
        model_kwargs: dict
    ) -> None:
        if not self.enabled:
            return

        wandb = self._get_wandb()
        wandb.init(
            project=self.project,
            name=model_name,
            group=f"{model_name}__lr_{config.learning_rate:g}",
            tags=[
                f"model_{model_name}",
                f"lr_{config.learning_rate:g}",
                f"seed_{config.seed}",
                f"ws_{config.window_size_model}",
                f"bs_{config.batch_size}",
            ],
            config={
                **config.to_dict(),
                "model": model_name,
                "model_kwargs": model_kwargs,
            }
        )

    def finish_run(self, final_loss: float, best_models: dict) -> None:
        if not self.enabled:
            return

        wandb = self._get_wandb()
        summary = {"final_loss": final_loss}

        for metric_name, info in best_models.items():
            if info.get("epoch") is not None:
                summary[f"best_val/{metric_name}"] = info["value"]
                summary[f"best_val_epoch/{metric_name}"] = info["epoch"]

        wandb.summary.update(summary)
        wandb.finish()

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None, prefix: str = "") -> None:
        if not self.enabled:
            return

        wandb = self._get_wandb()
        log_dict = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}
        if step is not None:
            wandb.log(log_dict, step=step)
        else:
            wandb.log(log_dict)

# endregion


# region Results Management
# ---------------------------------------------------------------------------
# Results Management
# ---------------------------------------------------------------------------

class ResultsManager:
    def __init__(self, checkpoint_dir: Path, test_output_dir: Path = None):
        self.checkpoint_dir = checkpoint_dir
        self.test_output_dir = test_output_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.test_output_dir:
            self.test_output_dir.mkdir(parents=True, exist_ok=True)

    def is_training_completed(self, run_id: str) -> bool:
        results_file = self.checkpoint_dir / run_id / "logs" / f"{run_id}_results.json"
        return results_file.exists()

    def is_test_completed(self, run_id: str, mode: str = "zeroshot") -> bool:
        if not self.test_output_dir:
            return False
        return (self.test_output_dir / mode / f"{run_id}_test_results.json").exists()

    def checkpoint_exists(self, run_id: str) -> bool:
        return (self.checkpoint_dir / run_id).exists()

    def get_checkpoint_dir(self, run_id: str) -> Path:
        return self.checkpoint_dir / run_id

    def get_logs_dir(self, run_id: str) -> Path:
        log_dir = self.get_checkpoint_dir(run_id) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def get_checkpoint_path(self, run_id: str, metric_name: str) -> Path:
        ckpt_dir = self.get_checkpoint_dir(run_id)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return ckpt_dir / f"best_{metric_name}.pt"

    def save_checkpoint(
        self,
        *,
        run_id: str,
        model: torch.nn.Module,
        epoch: int,
        metric_name: str,
        metric_value: float,
    ) -> Path:
        ckpt_path = self.get_checkpoint_path(run_id, metric_name)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metric_name": metric_name,
                "metric_value": metric_value,
            },
            ckpt_path,
        )
        return ckpt_path

    def load_checkpoint(
        self,
        *,
        run_id: str,
        model: torch.nn.Module,
        metric_name: str,
        device: torch.device,
    ) -> Dict[str, Any]:
        ckpt_path = self.get_checkpoint_path(run_id, metric_name)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=device)
        missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if missing:
            raise RuntimeError(f"Missing keys when loading {run_id}: {missing}")
        if unexpected:
            print(f"[WARN] Ignoring unexpected state_dict keys for {run_id}: {unexpected}")
        return checkpoint

    # ── few-shot checkpoints ───────────────────────────────────────
    def get_fewshot_checkpoint_path(
        self,
        run_id: str,
        metric_name: str,
        experiment: str,
        checkpoint_type: str = "best",
    ) -> Path:
        ckpt_dir = self.get_checkpoint_dir(run_id) / "fewshot"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        safe_name = experiment.replace("/", "_").replace("\\", "_")
        return ckpt_dir / f"fewshot_{checkpoint_type}_{metric_name}_{safe_name}.pt"

    def save_fewshot_checkpoint(
        self,
        *,
        run_id: str,
        state_dict: Dict[str, Any],
        epoch: int,
        metric_name: str,
        metric_value: float,
        experiment: str,
        checkpoint_type: str = "best",
    ) -> Path:
        ckpt_path = self.get_fewshot_checkpoint_path(
            run_id,
            metric_name,
            experiment,
            checkpoint_type=checkpoint_type,
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": state_dict,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "experiment": experiment,
                "checkpoint_type": checkpoint_type,
            },
            ckpt_path,
        )
        return ckpt_path

    def load_fewshot_checkpoint(
        self,
        *,
        run_id: str,
        model: torch.nn.Module,
        metric_name: str,
        experiment: str,
        device: torch.device,
        checkpoint_type: str = "best",
    ) -> Dict[str, Any]:
        ckpt_path = self.get_fewshot_checkpoint_path(
            run_id,
            metric_name,
            experiment,
            checkpoint_type=checkpoint_type,
        )
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Few-shot checkpoint not found at {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=device)
        missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if missing:
            raise RuntimeError(f"Missing keys when loading {run_id}: {missing}")
        if unexpected:
            print(f"[WARN] Ignoring unexpected state_dict keys for {run_id}: {unexpected}")
        return checkpoint

    def save_training_results(self, *, run_id: str, results: Dict[str, Any]) -> Path:
        import json

        log_dir = self.get_logs_dir(run_id)
        results_path = log_dir / f"{run_id}_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        return results_path

    def save_training_config(self, *, run_id: str, config: ExperimentConfig) -> None:
        log_dir = self.get_logs_dir(run_id)
        config_path = log_dir / f"{run_id}_config.json"
        with open(config_path, "w") as f:
            json.dump({"run_id": run_id, "config": config.to_dict()}, f, indent=2)

    def load_training_config(self, *, run_id: str) -> Tuple[ExperimentConfig, str]:
        log_dir = self.get_logs_dir(run_id)
        config_path = log_dir / f"{run_id}_config.json"
        with open(config_path, "r") as f:
            payload = json.load(f)
            config_data = payload["config"]
            stored_run_id = payload.get("run_id")

            valid_fields = {f.name for f in fields(ExperimentConfig)}
            unknown = set(config_data) - valid_fields
            if unknown:
                print(f"[WARN] Ignoring unknown config keys for {run_id}: {unknown}")
                config_data = {k: v for k, v in config_data.items() if k in valid_fields}
            config = ExperimentConfig(**config_data)
            config = _resolve_config_classes(config)

            import inspect
            model_sig = inspect.signature(config.model_class.__init__)
            valid_model_kwargs = set(model_sig.parameters)
            unknown_mk = set(config.model_kwargs) - valid_model_kwargs
            if unknown_mk:
                print(f"[WARN] Ignoring unknown model_kwargs for {run_id}: {unknown_mk}")
                config.model_kwargs = {
                    k: v for k, v in config.model_kwargs.items() if k in valid_model_kwargs
                }
            return config, stored_run_id

    def save_test_results(self, run_id: str, results: Dict[str, Any], mode: str = "zeroshot") -> Path:
        if not self.test_output_dir:
            raise ValueError("test_output_dir not configured")
        import json
        mode_dir = self.test_output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        output_file = mode_dir / f"{run_id}_test_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        return output_file

# endregion


# region Experiment Configuration
# ---------------------------------------------------------------------------
# Experiment Configuration
# ---------------------------------------------------------------------------

def _get_registry():
    from ..path import (
        TimeSeriesAffinePath,
        HilbertNaivePath,
        CylindricalGeodesicPath,
    )

    from os_flow_matching.dataset.transform import (
        Identity,
        ExtendedHilbert
    )
    
    from ..model import (
        TimeSeriesUNet,
        HilbertNaiveUNet,
        CylindricalGeodesicUNet,
    )

    return {
        "TimeSeriesAffine": (TimeSeriesAffinePath, TimeSeriesUNet, Identity),
        "HilbertNaive": (HilbertNaivePath, HilbertNaiveUNet, ExtendedHilbert),
        "CylindricalGeodesic": (CylindricalGeodesicPath, CylindricalGeodesicUNet, ExtendedHilbert),
        "Supervised": (None, TimeSeriesUNet, Identity),
    }


def _resolve_config_classes(config: ExperimentConfig) -> ExperimentConfig:
    registry = _get_registry()

    if config.model_name in registry:
        reg_path, reg_model, reg_transform = registry[config.model_name]
        config.path_class = reg_path
        config.model_class = reg_model
        config.transform_class = reg_transform

        if config.transform_name is not None:
            from os_flow_matching.dataset.transform import TRANSFORMS
            if config.transform_name not in TRANSFORMS:
                raise ValueError(
                    f"Unknown transform {config.transform_name!r}. "
                    f"Available: {list(TRANSFORMS.keys())}"
                )
            config.transform_class = TRANSFORMS[config.transform_name]

        return config
    else:
        raise ValueError(f"Unknown model name {config.model_name!r}")

def ensure_list(val: Union[int, float, Sequence[Union[int, float]]]) -> List[Union[int, float]]:
    if isinstance(val, (list, tuple)):
        return list(val)
    return [val]

@dataclass
class ExperimentConfig:
    seed: int
    batch_size: int
    window_size_model: int
    window_size_sampler: int
    learning_rate: float

    # Training parameters
    model_name: str
    path_class: Any
    model_class: Any
    transform_class: Any
    transform_name: str | None = None
    model_kwargs: Dict[str, Any] = field(default_factory=dict)

    num_epochs: int = 500
    validate_every: int = 10
    early_stopping_patience: int = 0
    # Consecutive validations without improvement before stopping; 0 disables

    root_dir: str = ""
    dataset_type: str | None = None
    source_spec: dict = field(default_factory=dict)
    target_spec: dict = field(default_factory=dict)
    num_workers: int = 64
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    condition_on_x0: bool = False

    # LR scheduler parameters
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.001

    # Regularization parameters
    weight_decay: float = 1e-3
    dropout: float = 0.0
    x0_noise_std: float = 0.0
    channel_weights: tuple[float, ...] | None = None
    psd_ratio_loss_weight: float = 0.0
    psd_ratio_band_hz: tuple[float, float] = (1.0, 39.0)
    psd_step_size: float | None = None
    # Gradient clipping
    grad_clip_threshold: float = 10.0

    # Test/evaluation settings
    num_steps: int = 80
    tracked_metrics: tuple = ()

    @property
    def is_supervised(self) -> bool:
        return self.path_class is None

    def make_run_id(self) -> str:
        base = f"{self.model_name}_seed{self.seed}_bs{self.batch_size}_ws{self.window_size_model}_lr{self.learning_rate}"
        if self.transform_name is not None:
            base += f"_tf{self.transform_name}"
        return base

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["path_class"] = self.path_class.__name__ if self.path_class else None
        d["model_class"] = self.model_class.__name__ if self.model_class else None
        d["transform_class"] = self.transform_class.__name__ if self.transform_class else None
        return d

    def get_sampler_kwargs(self) -> Dict[str, Any]:
        """Return all kwargs needed for sampler_factory()."""
        return {
            "root_dir": self.root_dir,
            "dataset_type": self.dataset_type,
            "source_spec": self.source_spec,
            "target_spec": self.target_spec,
            "num_workers": self.num_workers,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "window_size": self.window_size_sampler,
            "transform": self.transform_class(model_window_size=self.window_size_model),
        }


def build_experiment_grid(
        paths: List[dict],
        seeds: Union[int, Sequence[int]],
        batch_sizes: Union[int, Sequence[int]],
        window_sizes_sampler: Union[int, Sequence[int]] = None,
        window_sizes_model: Union[int, Sequence[int]] = None,
        learning_rates: Union[float, Sequence[float]] = None,
        *,
        transform_name: str | None = None,
        num_epochs: int = 500,
        validate_every: int = 10,
        early_stopping_patience: int = 0,
        root_dir: str = "",
        dataset_type: str | None = None,
        source_spec: dict = None,
        target_spec: dict = None,
        num_workers: int = 64,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        condition_on_x0: bool = False,
        dropout: float = 0.0,
        # LR scheduler parameters
        warmup_ratio: float = 0.05,
        min_lr_ratio: float = 0.001,
        # Regularization parameters
        weight_decay: float = 1e-3,
        x0_noise_std: float = 0.0,
        psd_ratio_loss_weight: float = 0.0,
        psd_ratio_band_hz: tuple[float, float] = (1.0, 39.0),
        psd_step_size: float | None = None,
        # Gradient clipping
        grad_clip_threshold: float = 10.0,
        # Test/evaluation settings
        num_steps: int = 80,
        tracked_metrics: tuple = (),
) -> List[ExperimentConfig]:
    configs = []

    seeds = [int(x) for x in ensure_list(seeds)]
    batch_sizes = [int(x) for x in ensure_list(batch_sizes)]
    window_sizes_sampler_list = [int(x) for x in ensure_list(window_sizes_sampler)]
    window_sizes_model_list = [int(x) for x in ensure_list(window_sizes_model)]
    learning_rates = [float(x) for x in ensure_list(learning_rates)]

    flow_configs = build_flow_configs(paths, transform_name=transform_name)
    items = product(flow_configs, seeds, batch_sizes, window_sizes_sampler_list, window_sizes_model_list, learning_rates)
    for (name, PClass, MClass, TClass, m_kwargs, t_name, item_cw), seed, bs, ws_sampler, ws_model, lr in items:
        inferred_kwargs = infer_model_kwargs_from_specs(
            MClass,
            source_spec or {},
            condition_on_x0=condition_on_x0,
        )
        model_kwargs = {**inferred_kwargs, **(m_kwargs or {}), "dropout": dropout}
        configs.append(
            ExperimentConfig(
                model_name=name,
                path_class=PClass,
                model_class=MClass,
                transform_class=TClass,
                transform_name=t_name,
                model_kwargs=model_kwargs,
                seed=seed,
                batch_size=bs,
                window_size_model=ws_model,
                window_size_sampler=ws_sampler,
                learning_rate=lr,
                num_epochs=num_epochs,
                validate_every=validate_every,
                early_stopping_patience=early_stopping_patience,
                root_dir=root_dir,
                dataset_type=dataset_type,
                source_spec=source_spec or {},
                target_spec=target_spec or {},
                num_workers=num_workers,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                condition_on_x0=condition_on_x0,
                # LR scheduler parameters
                warmup_ratio=warmup_ratio,
                min_lr_ratio=min_lr_ratio,
                # Regularization parameters
                weight_decay=weight_decay,
                dropout=dropout,
                x0_noise_std=x0_noise_std,
                channel_weights=item_cw,
                psd_ratio_loss_weight=psd_ratio_loss_weight,
                psd_ratio_band_hz=psd_ratio_band_hz,
                psd_step_size=psd_step_size,
                # Gradient clipping
                grad_clip_threshold=grad_clip_threshold,
                # Test/evaluation settings
                num_steps=num_steps,
                tracked_metrics=tracked_metrics,
            )
        )
    return configs


def build_flow_configs(sweep, transform_name: str | None = None):
    configs = []
    REGISTRY = _get_registry()
    for item in sweep:
        method_id = item["id"]
        try:
            PathClass, ModelClass, TransformClass = REGISTRY[method_id]
        except KeyError as e:
            known = ", ".join(sorted(REGISTRY.keys()))
            raise KeyError(f"Unknown sweep id {method_id!r}. Known ids: {known}") from e

        # Per-item "transform" overrides global transform_name
        item_transform = item.get("transform", transform_name)
        if item_transform is not None:
            from os_flow_matching.dataset.transform import TRANSFORMS
            if item_transform not in TRANSFORMS:
                raise ValueError(
                    f"Unknown transform {item_transform!r}. "
                    f"Available: {list(TRANSFORMS.keys())}"
                )
            TransformClass = TRANSFORMS[item_transform]

        model_kwargs = item.get("model_kwargs", {})
        item_channel_weights = item.get("channel_weights", None)
        configs.append((method_id, PathClass, ModelClass, TransformClass, model_kwargs, item_transform, item_channel_weights))
    return configs


# endregion


# region Printing Utilities
# ---------------------------------------------------------------------------
# Printing Utilities
# ---------------------------------------------------------------------------

def print_sweep_summary(
    configs: List[ExperimentConfig],
    *,
    action: str = "run",
) -> None:
    seeds = sorted(set(c.seed for c in configs))
    batch_sizes = sorted(set(c.batch_size for c in configs))
    window_sizes_model = sorted(set(c.window_size_model for c in configs))
    window_sizes_sampler = sorted(set(c.window_size_sampler for c in configs))
    learning_rates = sorted(set(c.learning_rate for c in configs))
    model_names = sorted(set(c.model_name for c in configs))

    print(f"Total experiments to {action}: {len(configs)}")
    print(f"  - Seeds: {seeds}")
    print(f"  - Batch sizes: {batch_sizes}")
    print(f"  - Window sizes (sampler): {window_sizes_sampler}")
    print(f"  - Window sizes (model): {window_sizes_model}")
    print(f"  - Learning rates: {learning_rates}")
    print(f"  - Models: {model_names}")


def print_experiment_header(
    experiment_num: int,
    total: int,
    run_id: str,
    title: str = None,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"Experiment {experiment_num}/{total}" + (f": {run_id}" if not title else ""))
    if title:
        print(title)
    print(f"{'=' * 60}")
# endregion
