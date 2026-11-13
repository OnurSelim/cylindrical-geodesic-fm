from __future__ import annotations

from pathlib import Path
from typing import Any, Final

# ---------------------------------------------------------------------------- #
#                            Configuration Selection                           #
# ---------------------------------------------------------------------------- #
# Available profiles: "scg_time_series", "scg_beat", "ppg_time_series"
ACTIVE_CONFIG: Final[str] = "scg_beat"


# ---------------------------------------------------------------------------- #
#                              Base Configuration                              #
# ---------------------------------------------------------------------------- #
# _BASE holds settings shared by every profile. Anything that varies per
# profile lives in _PROFILES below.
_BASE: dict[str, Any] = {
    # Signal
    "FS": 500,
    # Dataset
    "NUM_WORKERS": 64,
    "VAL_RATIO": 0.25,
    "TEST_RATIO": 0.25,
    # Training
    "USE_WANDB": True,
    "WINDOW_SIZE_SAMPLER": [4096],
    "WINDOW_SIZE_MODEL": [2048],
    "TRACKED_METRICS": ("l2_mean", "dtw_mean", "ht_intrinsic_mean"),
    # Stop training after this many consecutive validations without any tracked
    # metric improving. 0 disables early stopping.
    "EARLY_STOPPING_PATIENCE": 5,
    # Test
    "NUM_STEPS": 100,
    "TEST_BEST_MODEL_METRICS": ["l2_mean", "ht_intrinsic_mean", "dtw_mean"],
    # LR scheduler
    "WARMUP_RATIO": 0.01,
    "MIN_LR_RATIO": 0.001,
    # Optimizer
    "WEIGHT_DECAY": 1e-2,
    # Dropout
    "DROPOUT": 0.05,
    # Noise injection to x0 conditioning
    "CONDITION_ON_X0": True,
    "X0_NOISE_STD": 0.05,
    # PSD auxiliary loss for Flow Matching
    "PSD_RATIO_LOSS_WEIGHT": 0,
    "PSD_RATIO_BAND_HZ": (0, 250),
    "PSD_STEP_SIZE": None,
    # Gradient clipping
    "GRAD_CLIP_THRESHOLD": 1.0,
}


# ---------------------------------------------------------------------------- #
#                       Shared Path × Transform Matrix                         #
# ---------------------------------------------------------------------------- #
# Full Hilbert path - transform sweep. HilbertNaive outputs 2 channels
# (real, imag); Cylindrical Geodesic outputs 3 channels.
_FULL_PATHS: list[dict[str, Any]] = [
    {"id": "HilbertNaive",        "transform": "hilbert",          "channel_weights": (1.0, 1.0)},
    {"id": "HilbertNaive",        "transform": "extended_hilbert", "channel_weights": (1.0, 1.0)},
    {"id": "CylindricalGeodesic", "transform": "hilbert",          "channel_weights": (1.0, 1.0, 1.0)},
    {"id": "CylindricalGeodesic", "transform": "extended_hilbert", "channel_weights": (1.0, 1.0, 1.0)},
    {"id": "Supervised"},
    {"id": "TimeSeriesAffine"},
]


# ---------------------------------------------------------------------------- #
#                                   Profiles                                   #
# ---------------------------------------------------------------------------- #
_PROFILES: dict[str, dict[str, Any]] = {
    "ppg_time_series": {
        **_BASE,
        "ROOT_DIR": "data/os_dataset_6b1d65a",
        "SOURCE_SPEC": {
            "Sternum": {"src_mod": ["PPG"], "cnd_mod": ["ECG"]},
        },
        "TARGET_SPEC": {
            "Biopac": ["PPG"],
        },
        "PATHS": _FULL_PATHS,
        "NUM_EPOCHS": 5000,
        "VALIDATE_EVERY": 200,
        "BATCH_SIZE": [512],
        "LEARNING_RATE": [3e-4],
        "SEED": [6, 10, 11, 13],
        "PSD_RATIO_LOSS_WEIGHT": 0.05,
        "PSD_RATIO_BAND_HZ": (1.0, 4.0),
        "PSD_STEP_SIZE": 0.1,
    },
    "scg_beat": {
        **_BASE,
        "ROOT_DIR": "data/os_dataset_069d3ac",
        "SOURCE_SPEC": {
            "Clavicle": {"src_mod": ["SCG"], "cnd_mod": ["ECG"]},
        },
        "TARGET_SPEC": {
            "Sternum": ["SCG"],
        },
        "PATHS": _FULL_PATHS,
        "NUM_EPOCHS": 5000,
        "VALIDATE_EVERY": 200,
        "BATCH_SIZE": [4096],
        "WINDOW_SIZE_SAMPLER": [1024],  # full stored beat; the beat sampler doesn't crop
        "WINDOW_SIZE_MODEL": [512],
        "LEARNING_RATE": [3e-4],
        "SEED": [6, 10, 11, 13],
        "VAL_RATIO": 0.25,
        "TEST_RATIO": 0.25,
        "PSD_RATIO_LOSS_WEIGHT": 0.05,
        "PSD_RATIO_BAND_HZ": (1.0, 39.0),
        "PSD_STEP_SIZE": 0.1,
    },
}

DATASET_TYPE_REG = {
    "time_series": ["data/os_dataset_6b1d65a"],
    "beat": ["data/os_dataset_069d3ac"],
}


def _resolve_dataset_type(root_dir: str) -> str:
    for dtype, roots in DATASET_TYPE_REG.items():
        if root_dir in roots:
            return dtype
    raise ValueError(f"Could not resolve dataset type for root_dir '{root_dir}'. Please check the configuration.")

_cfg = _PROFILES[ACTIVE_CONFIG]

# Signal settings
FS: Final[int] = _cfg["FS"]

# Dataset settings
ROOT_DIR: Final[str] = _cfg["ROOT_DIR"]
DATASET_TYPE: Final[str] = _resolve_dataset_type(ROOT_DIR)
NUM_WORKERS: Final[int] = _cfg["NUM_WORKERS"]
VAL_RATIO: Final[float] = _cfg["VAL_RATIO"]
TEST_RATIO: Final[float] = _cfg["TEST_RATIO"]
SOURCE_SPEC: Final[dict[str, dict[str, list[str]]]] = _cfg["SOURCE_SPEC"]
TARGET_SPEC: Final[dict[str, list[str]]] = _cfg["TARGET_SPEC"]

# Training settings
CONDITION_ON_X0: Final[bool] = _cfg["CONDITION_ON_X0"]
PATHS: Final[list[dict[str, object]]] = _cfg["PATHS"]

_TARGET_SIGNAL = next(iter(next(iter(TARGET_SPEC.values())))).lower()
_dataset_id = Path(ROOT_DIR).name.split("_")[-1]
CHECKPOINT_DIR: Final[Path] = Path(f"results/{_TARGET_SIGNAL}_{_dataset_id}/model_checkpoints")

NUM_EPOCHS: Final[int] = _cfg["NUM_EPOCHS"]
VALIDATE_EVERY: Final[int] = _cfg["VALIDATE_EVERY"]
EARLY_STOPPING_PATIENCE: Final[int] = _cfg["EARLY_STOPPING_PATIENCE"]
USE_WANDB: Final[bool] = _cfg["USE_WANDB"]
WANDB_PROJECT: Final[str] = f"fm-{_TARGET_SIGNAL}-{_dataset_id}"

BATCH_SIZE: Final[list[int]] = _cfg["BATCH_SIZE"]

WINDOW_SIZE_SAMPLER: Final[list[int]] = _cfg["WINDOW_SIZE_SAMPLER"]
WINDOW_SIZE_MODEL: Final[list[int]] = _cfg["WINDOW_SIZE_MODEL"]

LEARNING_RATE: Final[list[float]] = _cfg["LEARNING_RATE"]
SEED: Final[list[int]] = _cfg["SEED"]

TRACKED_METRICS: Final[tuple[str, ...]] = _cfg["TRACKED_METRICS"]

# Test settings
NUM_STEPS: Final[int] = _cfg["NUM_STEPS"]
TEST_BEST_MODEL_METRICS: Final[list[str]] = _cfg["TEST_BEST_MODEL_METRICS"]

# LR scheduler
WARMUP_RATIO: Final[float] = _cfg["WARMUP_RATIO"]
MIN_LR_RATIO: Final[float] = _cfg["MIN_LR_RATIO"]

# Optimizer
WEIGHT_DECAY: Final[float] = _cfg["WEIGHT_DECAY"]

# Dropout
DROPOUT: Final[float] = _cfg["DROPOUT"]

# Noise injection to x0 conditioning
X0_NOISE_STD: Final[float] = _cfg["X0_NOISE_STD"]
PSD_RATIO_LOSS_WEIGHT: Final[float] = _cfg["PSD_RATIO_LOSS_WEIGHT"]
PSD_RATIO_BAND_HZ: Final[tuple[float, float]] = tuple(float(v) for v in _cfg["PSD_RATIO_BAND_HZ"])
PSD_STEP_SIZE: Final[float | None] = (
    None if _cfg["PSD_STEP_SIZE"] is None else float(_cfg["PSD_STEP_SIZE"])
)

# Gradient clipping
GRAD_CLIP_THRESHOLD: Final[float] = _cfg["GRAD_CLIP_THRESHOLD"]
