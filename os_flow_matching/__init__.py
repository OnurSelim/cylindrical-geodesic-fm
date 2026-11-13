from __future__ import annotations

import torch

from .utils import (
    build_experiment_grid,
    ExperimentConfig,
    ResultsManager,
    WandbManager,
    print_experiment_header,
    print_sweep_summary
)
from .runners import (
    run_training_experiment,
    run_fewshot_experiment,
    run_test_and_cache,
    print_final_summary,
    zeroshot_dir,
    fewshot_dir,
    tensors_cached,
    session_ranges_cached,
    has_fewshot_checkpoints,
)
from .factories import sampler_factory, trainer_factory


def get_device(gpu_id: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


