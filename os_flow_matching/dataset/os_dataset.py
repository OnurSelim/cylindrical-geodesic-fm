from __future__ import annotations

import json, os, re, random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


SessionPath = Path
DeviceName  = str
Modality = str

_subj_regex = re.compile(r"^(S\d{3})_Day\d+")


def _zscore(arr: np.ndarray, ignore_nan: bool = False, eps: float = 1e-12) -> np.ndarray:
    if ignore_nan:
        mean = np.nanmean(arr, axis=1, keepdims=True)
        std  = np.nanstd (arr, axis=1, keepdims=True)
    else:
        mean = arr.mean(axis=1, keepdims=True)
        std  = arr.std (axis=1, keepdims=True)

    return (arr - mean) / (std + eps)


def scan_tree(root: str) -> Dict[str, List[SessionPath]]:
    out: dict[str, list[SessionPath]] = defaultdict(list)
    with os.scandir(root) as it:
        for e in sorted(it, key=lambda x: x.name):
            if e.is_dir():
                m = _subj_regex.match(e.name)
                if m:
                    out[m.group(1)].append(Path(e.path))
    return dict(out)


@dataclass(frozen=True)
class _DevView:
    dev: str
    idxs: np.ndarray


class _SessionCache:
    def __init__(self, session_path: SessionPath):
        self.path = session_path
        self.arrays  : dict[str, np.ndarray]    = {}
        self.devices : dict[str, dict[str,int]] = {}
        self.metadata: dict[str, dict]           = {}
        self._load()

    def has(self, dev: DeviceName, modality: Modality) -> bool:
        return (dev in self.devices) and (modality in self.devices[dev])

    def num_samples(self, dev: DeviceName) -> int:
        return self.arrays[dev].shape[0]

    def slice_into(self, out: np.ndarray, views: list[_DevView], start: int, length: int) -> None:
        o = 0
        for v in views:
            blk = self.arrays[v.dev][start:start+length, v.idxs]  # (length, n_ch)
            out[o:o+blk.shape[1]] = blk.T
            o += blk.shape[1]

    def _file(self, dev: DeviceName, ext: str) -> Path:
        return self.path / f"{dev}{ext}"

    def _load(self) -> None:
        for fname in os.listdir(self.path):
            if not fname.endswith("_columns.json"):
                continue
            dev = fname[:-len("_columns.json")]
            with self._file(dev, "_columns.json").open("r") as j:
                data = json.load(j)
            
            if not isinstance(data, dict) or "signals" not in data:
                raise ValueError(
                    f"Invalid columns JSON format in {self._file(dev, '_columns.json')}: "
                    "expected a dict with a 'signals' key"
                )
            names = data["signals"]
            
            self.devices[dev] = {n: i for i, n in enumerate(names)}
            self.metadata[dev] = {
                k: v for k, v in data.items() if k != "signals"
            }
            arr = np.load(self._file(dev, ".npy"))
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32, copy=False)
            if isinstance(arr, np.memmap):
                arr = np.asarray(arr)
            self.arrays[dev] = arr


class TSWindowDataset(Dataset):
    def __init__(self, session_paths: Sequence[SessionPath],
                 source_spec: Dict[str, Dict[str, Sequence[str]]],
                 target_spec: Dict[str, Sequence[str]],
                 window_size: int = 2048,
                 step: int | None = None):

        self.source_spec = {d: tuple(spec.get("src_mod", ())) for d, spec in source_spec.items()}
        self.cond_spec   = {d: tuple(spec.get("cnd_mod", ())) for d, spec in source_spec.items()}
        self.target_spec = {d: tuple(chs) for d, chs in target_spec.items()}

        self.window_size = int(window_size)
        self.step        = int(step) if step is not None else int(window_size)

        self._n_src  = sum(len(ch) for ch in self.source_spec.values())
        self._n_cond = sum(len(ch) for ch in self.cond_spec.values())
        self._n_tgt  = sum(len(ch) for ch in self.target_spec.values())

        src_windows  : list[torch.Tensor] = []
        cond_windows : list[torch.Tensor] = []
        tgt_windows  : list[torch.Tensor] = []
        session_ranges: list[dict[str, object]] = []

        total_windows = 0

        for sess in sorted(session_paths):
            cache = _SessionCache(sess)
            sess_start = total_windows

            if not self._session_has_required(cache):
                continue

            src_views  = self._compute_views(cache, self.source_spec)
            tgt_views  = self._compute_views(cache, self.target_spec)

            cond_views, cond_missing = self._compute_views_for_cond(cache, self.cond_spec)

            used_devs = set(self.source_spec) | set(self.target_spec)
            L = min(cache.num_samples(d) for d in used_devs)
            if L < self.window_size:
                continue

            for start in range(0, L - self.window_size + 1, self.step):
                src = np.empty((self._n_src, self.window_size), dtype=np.float32)
                cache.slice_into(src, src_views, start, self.window_size)
                src = _zscore(src)

                if self._n_cond:
                    cond = np.full((self._n_cond, self.window_size),
                                   np.nan, dtype=np.float32)
                    o = 0
                    for view, miss_mask in zip(cond_views, cond_missing):
                        K = miss_mask.size
                        if view.idxs.size:
                            blk = cache.arrays[view.dev][start:start + self.window_size,
                                  view.idxs].T
                            write_pos = np.nonzero(~miss_mask)[0]
                            cond[o + write_pos] = blk
                        o += K

                    cond = _zscore(cond, ignore_nan=True)
                else:
                    cond = np.empty((0, self.window_size), dtype=np.float32)

                tgt = np.empty((self._n_tgt, self.window_size), dtype=np.float32)
                cache.slice_into(tgt, tgt_views, start, self.window_size)
                tgt = _zscore(tgt)

                src_windows.append(torch.from_numpy(src))
                cond_windows.append(torch.from_numpy(cond))
                tgt_windows.append(torch.from_numpy(tgt))
                total_windows += 1

            if total_windows > sess_start:
                session_ranges.append(
                    {
                        "session": sess,
                        "experiment": sess.stem,
                        "start": sess_start,
                        "end": total_windows,
                    }
                )

        if total_windows == 0:
            raise RuntimeError("No windows available with the given specs!")

        self._src  = torch.stack(src_windows,  dim=0)
        self._cond = torch.stack(cond_windows, dim=0) if self._n_cond else torch.empty((len(tgt_windows), 0, self.window_size))
        self._tgt  = torch.stack(tgt_windows,  dim=0)
        self._session_ranges = session_ranges
        self._sessions = [item["session"] for item in session_ranges]

    @staticmethod
    def _compute_views(cache: _SessionCache, spec: dict[str, tuple[str, ...]]) -> list[_DevView]:
        views: list[_DevView] = []
        for dev, chans in spec.items():
            cols = cache.devices[dev]
            idxs = np.fromiter((cols[c] for c in chans), dtype=np.int32, count=len(chans))
            views.append(_DevView(dev=dev, idxs=idxs))
        return views

    @staticmethod
    def _compute_views_for_cond(cache: _SessionCache, spec: dict[str, tuple[str, ...]]
                                ) -> tuple[list[_DevView], list[np.ndarray]]:
        views: list[_DevView] = []
        miss:  list[np.ndarray] = []
        for dev, chans in spec.items():
            cols = cache.devices.get(dev, {})
            present_idxs: list[int] = []
            missing_mask = np.zeros(len(chans), dtype=bool)
            for i, ch in enumerate(chans):
                if ch in cols:
                    present_idxs.append(cols[ch])
                else:
                    missing_mask[i] = True
            views.append(_DevView(dev=dev, idxs=np.array(present_idxs, dtype=np.int32)))
            miss.append(missing_mask)
        return views, miss

    def _session_has_required(self, cache: _SessionCache) -> bool:
        for dev, chans in list(self.source_spec.items()) + list(self.target_spec.items()):
            for ch in chans:
                if not cache.has(dev, ch):
                    return False
        return True

    def __len__(self) -> int:
        return self._tgt.shape[0]

    def __getitem__(self, idx: int):
        return {
            "s": self._src[idx],
            "c": self._cond[idx],
            "t": self._tgt[idx],
        }


class BeatWindowDataset(Dataset):
    def __init__(self, session_paths: Sequence[SessionPath],
                 source_spec: Dict[str, Dict[str, Sequence[str]]],
                 target_spec: Dict[str, Sequence[str]]):
        
        self.source_spec = {d: tuple(spec.get("src_mod", ())) for d, spec in source_spec.items()}
        self.cnd_spec   = {d: tuple(spec.get("cnd_mod", ())) for d, spec in source_spec.items()}
        self.target_spec = {d: tuple(chs) for d, chs in target_spec.items()}
        
        self._n_src  = sum(len(ch) for ch in self.source_spec.values())
        self._n_cond = sum(len(ch) for ch in self.cnd_spec.values())
        self._n_tgt  = sum(len(ch) for ch in self.target_spec.values())
        
        # Per-beat metadata field names (numeric stored as tensors, string as lists)
        _NUMERIC_META = ("ao_points", "ac_points", "beat_indices", "original_beat_lens")
        _STRING_META  = ("events",)

        src_beats  : list[torch.Tensor] = []
        cond_beats : list[torch.Tensor] = []
        tgt_beats  : list[torch.Tensor] = []
        session_ranges: list[dict[str, object]] = []
        beat_len: int = 0

        # Per-beat experiment label (session name, e.g. "S009_Day1")
        experiment_labels: list[str] = []

        # Metadata accumulators – keyed by device then field
        meta_numeric: dict[str, dict[str, list[int]]] = {
            d: {f: [] for f in _NUMERIC_META} for d in (set(source_spec) | set(target_spec))
        }
        meta_string: dict[str, dict[str, list[str]]] = {
            d: {f: [] for f in _STRING_META} for d in (set(source_spec) | set(target_spec))
        }

        for sess in sorted(session_paths):
            cache = _SessionCache(sess)
            sess_start = len(src_beats)
            
            if not self._session_has_required(cache):
                continue
            
            src_views  = self._compute_views(cache, self.source_spec)
            tgt_views  = self._compute_views(cache, self.target_spec)
            cond_views, cond_missing = self._compute_views_for_cond(cache, self.cnd_spec)

            used_devs = set(self.source_spec) | set(self.target_spec)
            n_beats_list = [cache.num_samples(d) for d in used_devs]
            if not n_beats_list:
                continue
            
            n_beats = min(n_beats_list)
            beat_len = cache.arrays[list(used_devs)[0]].shape[1]

            for beat_idx in range(n_beats):
                src = np.empty((self._n_src, beat_len), dtype=np.float32)
                o = 0
                for view in src_views:
                    blk = cache.arrays[view.dev][beat_idx, :, view.idxs]  # shape: (n_ch, beat_len) or (beat_len,)
                    if blk.ndim == 1:
                        blk = blk[np.newaxis, :]  # (1, beat_len)
                    n_ch = blk.shape[0]
                    src[o:o+n_ch] = blk  # (n_ch, beat_len)
                    o += n_ch
                src = _zscore(src)
                
                # Extract conditioning channels for this beat
                if self._n_cond:
                    cond = np.full((self._n_cond, beat_len), np.nan, dtype=np.float32)
                    o = 0
                    for view, miss_mask in zip(cond_views, cond_missing):
                        K = miss_mask.size
                        if view.idxs.size:
                            blk = cache.arrays[view.dev][beat_idx, :, view.idxs]  # (n_ch, beat_len) or (beat_len,)
                            if blk.ndim == 1:
                                blk = blk[np.newaxis, :]  # (1, beat_len)
                            write_pos = np.nonzero(~miss_mask)[0]
                            cond[o + write_pos] = blk
                        o += K
                    cond = _zscore(cond, ignore_nan=True)
                else:
                    cond = np.empty((0, beat_len), dtype=np.float32)
                
                # Extract target channels for this beat
                tgt = np.empty((self._n_tgt, beat_len), dtype=np.float32)
                o = 0
                for view in tgt_views:
                    blk = cache.arrays[view.dev][beat_idx, :, view.idxs]  # (n_ch, beat_len) or (beat_len,)
                    if blk.ndim == 1:
                        blk = blk[np.newaxis, :]  # (1, beat_len)
                    n_ch = blk.shape[0]
                    tgt[o:o+n_ch] = blk  # (n_ch, beat_len)
                    o += n_ch
                tgt = _zscore(tgt)
                
                src_beats.append(torch.from_numpy(src))
                cond_beats.append(torch.from_numpy(cond))
                tgt_beats.append(torch.from_numpy(tgt))
                experiment_labels.append(sess.name)

                # Collect per-beat metadata for each device
                for dev in used_devs:
                    dm = cache.metadata.get(dev, {})
                    for f in _NUMERIC_META:
                        vals = dm.get(f)
                        meta_numeric[dev][f].append(vals[beat_idx] if vals is not None else -1)
                    for f in _STRING_META:
                        vals = dm.get(f)
                        meta_string[dev][f].append(vals[beat_idx] if vals is not None else "")

            sess_end = len(src_beats)
            if sess_end > sess_start:
                session_ranges.append(
                    {
                        "session": sess,
                        "experiment": sess.name,
                        "start": sess_start,
                        "end": sess_end,
                    }
                )
        
        if not src_beats:
            raise RuntimeError("No beats available with the given specs!")
        
        self._src  = torch.stack(src_beats,  dim=0)
        self._cond = torch.stack(cond_beats, dim=0) if self._n_cond else torch.empty((len(tgt_beats), 0, beat_len))
        self._tgt  = torch.stack(tgt_beats,  dim=0)
        
        # Store per-beat metadata: numeric fields as long tensors, string fields as lists
        self._meta_numeric: dict[str, dict[str, torch.Tensor]] = {
            dev: {f: torch.tensor(vals, dtype=torch.long) for f, vals in fields.items()}
            for dev, fields in meta_numeric.items()
        }
        self._meta_string: dict[str, dict[str, list[str]]] = meta_string
        self._experiment_labels: list[str] = experiment_labels

        self.beat_len = beat_len
        self._session_ranges = session_ranges
        self._sessions = [item["session"] for item in session_ranges]


    @staticmethod
    def _compute_views(cache: _SessionCache, spec: dict[str, tuple[str, ...]]) -> list[_DevView]:
        views: list[_DevView] = []
        for dev, chans in spec.items():
            cols = cache.devices[dev]
            idxs = np.fromiter((cols[c] for c in chans), dtype=np.int32, count=len(chans))
            views.append(_DevView(dev=dev, idxs=idxs))
        return views
    
    @staticmethod
    def _compute_views_for_cond(cache: _SessionCache, spec: dict[str, tuple[str, ...]]
                                ) -> tuple[list[_DevView], list[np.ndarray]]:
        views: list[_DevView] = []
        miss:  list[np.ndarray] = []
        for dev, chans in spec.items():
            cols = cache.devices.get(dev, {})
            present_idxs: list[int] = []
            missing_mask = np.zeros(len(chans), dtype=bool)
            for i, ch in enumerate(chans):
                if ch in cols:
                    present_idxs.append(cols[ch])
                else:
                    missing_mask[i] = True
            views.append(_DevView(dev=dev, idxs=np.array(present_idxs, dtype=np.int32)))
            miss.append(missing_mask)
        return views, miss
    
    def _session_has_required(self, cache: _SessionCache) -> bool:
        for dev, chans in list(self.source_spec.items()) + list(self.target_spec.items()):
            for ch in chans:
                if not cache.has(dev, ch):
                    return False
        return True
    
    def __len__(self) -> int:
        return self._tgt.shape[0]
    
    def __getitem__(self, idx: int):
        item = {
            "s": self._src[idx],
            "c": self._cond[idx],
            "t": self._tgt[idx],
        }
        # Attach per-beat metadata (numeric as tensors, string by index)
        for dev, fields in self._meta_numeric.items():
            for f, tensor in fields.items():
                item[f"meta_{dev}_{f}"] = tensor[idx]
        for dev, fields in self._meta_string.items():
            for f, vals in fields.items():
                item[f"meta_{dev}_{f}"] = vals[idx]
        item["meta_experiment"] = self._experiment_labels[idx]
        return item

    def get_metadata_at(self, indices: Sequence[int]) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        meta["experiment"] = [self._experiment_labels[i] for i in indices] if indices else []
        for dev, fields in self._meta_numeric.items():
            for f, tensor in fields.items():
                meta[f"{dev}_{f}"] = tensor[indices].tolist() if len(indices) else []
        for dev, fields in self._meta_string.items():
            for f, vals in fields.items():
                meta[f"{dev}_{f}"] = [vals[i] for i in indices] if indices else []
        return meta


def train_val_test_split(subject_to_sessions : Dict[str, List[SessionPath]],
                         val_ratio : float = 0.1,
                         test_ratio: float = 0.1,
                         seed      : int   = 0) -> Tuple[List[Path], List[Path], List[Path]]:
    random.seed(seed)
    subjects = list(subject_to_sessions)
    random.shuffle(subjects)

    n = len(subjects)
    n_test = int(round(n * test_ratio))
    n_val  = int(round(n * val_ratio))

    test_sub  = subjects[:n_test]
    val_sub   = subjects[n_test:n_test+n_val]
    train_sub = subjects[n_test+n_val:]

    coll = lambda sub_list: sum((subject_to_sessions[s] for s in sub_list), [])
    return coll(train_sub), coll(val_sub), coll(test_sub)
