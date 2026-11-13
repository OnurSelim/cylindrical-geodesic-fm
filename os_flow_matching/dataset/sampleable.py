from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List, Union, Sequence

import torch
from torch import nn

from .os_dataset import TSWindowDataset, BeatWindowDataset, scan_tree, train_val_test_split


class Sampleable(ABC):
    batch_size: int

    @abstractmethod
    def sample(
        self,
        split: str = "train"
    ):
        pass

    @abstractmethod
    def sample_at(
        self,
        indices: Union[int, Sequence[int], torch.Tensor],
        split: str = "train"
    ):
        pass

    @abstractmethod
    def dataset_size(
        self,
        split: str
    ) -> int:
        pass


class _BaseDatasetSampler(nn.Module, Sampleable, ABC):
    def _initialize_runtime(
        self,
        *,
        split: str,
        batch_size: int,
        num_workers: int,
        prefetch_factor: Optional[int],
        pin_memory: bool,
    ) -> None:
        self.batch_size = batch_size
        self._is_shutdown = False

        if split not in self.datasets:
            raise ValueError(f"split has to be one of {list(self.datasets)}")

        self._loaders: Dict[str, torch.utils.data.DataLoader] = {}
        self._iters: Dict[str, Optional[object]] = {}

        for sp, ds in self.datasets.items():
            loader_kwargs = {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "persistent_workers": (num_workers > 0),
                "shuffle": (sp == "train"),
            }
            if num_workers > 0 and prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = prefetch_factor

            loader = torch.utils.data.DataLoader(ds, **loader_kwargs)
            self._loaders[sp] = loader
            self._iters[sp] = iter(loader)

        loaders_list = list(self._loaders.values())
        self._finalizer = weakref.finalize(self, _BaseDatasetSampler._cleanup_all, loaders_list)

    def _validate_split(self, split: str) -> None:
        if split not in self.datasets:
            raise ValueError(f"split has to be one of {list(self.datasets)}")

    def _ensure_active(self) -> None:
        if self._is_shutdown:
            raise RuntimeError("Sampler has been shut down and can no longer be used.")

    @staticmethod
    def _normalize_indices(indices: Union[int, Sequence[int], torch.Tensor]) -> list[int]:
        if isinstance(indices, int):
            return [indices]
        if isinstance(indices, torch.Tensor):
            return [int(i) for i in indices.tolist()]
        return [int(i) for i in indices]

    @staticmethod
    def _validate_indices(indices: Sequence[int], n: int) -> None:
        for i in indices:
            if not 0 <= i < n:
                raise IndexError(f"index {i} out of range (dataset size = {n})")

    def _next_batch(self, split: str) -> Dict[str, torch.Tensor]:
        self._ensure_active()
        self._validate_split(split)

        loader = self._loaders[split]
        it = self._iters[split]
        if it is None:
            raise RuntimeError("Sampler iterator is unavailable. Was shutdown() called?")

        try:
            return next(it)
        except StopIteration:
            it = iter(loader)
            self._iters[split] = it
            return next(it)

    @abstractmethod
    def _make_empty_cond(
        self,
        *,
        batch_len: int,
        dataset: Any,
        ref_sample: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        pass

    @torch.no_grad()
    def sample(
        self,
        split: str = "train"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = self._next_batch(split)
        src = batch['s'].to(self.dummy, non_blocking=True)
        cond = batch['c'].to(self.dummy, non_blocking=True)
        tgt = batch['t'].to(self.dummy, non_blocking=True)
        return self._apply_transform(src, cond, tgt)

    @torch.no_grad()
    def sample_at(
        self,
        indices: Union[int, Sequence[int], torch.Tensor],
        split: str = "train"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ensure_active()
        self._validate_split(split)
        dataset = self.datasets[split]

        normalized_indices = self._normalize_indices(indices)
        n = len(dataset)
        self._validate_indices(normalized_indices, n)

        if not normalized_indices:
            if n == 0:
                raise RuntimeError("Cannot sample from an empty dataset.")
            ref_sample = dataset[0]
            src = torch.empty((0, *ref_sample['s'].shape),
                              device=self.dummy.device,
                              dtype=ref_sample['s'].dtype)
            tgt = torch.empty((0, *ref_sample['t'].shape),
                              device=self.dummy.device,
                              dtype=ref_sample['t'].dtype)
            if ref_sample['c'].numel():
                cond = torch.empty((0, *ref_sample['c'].shape),
                                   device=self.dummy.device,
                                   dtype=ref_sample['c'].dtype)
            else:
                cond = self._make_empty_cond(batch_len=0, dataset=dataset, ref_sample=ref_sample)
            return src, cond, tgt

        batch = [dataset[i] for i in normalized_indices]
        src = torch.stack([b['s'] for b in batch]).to(self.dummy, non_blocking=True)
        tgt = torch.stack([b['t'] for b in batch]).to(self.dummy, non_blocking=True)

        if batch[0]['c'].numel():
            cond = torch.stack([b['c'] for b in batch]).to(self.dummy, non_blocking=True)
        else:
            cond = self._make_empty_cond(batch_len=len(normalized_indices),
                                         dataset=dataset,
                                         ref_sample=batch[0])

        return self._apply_transform(src, cond, tgt)

    def _apply_transform(
        self,
        src: torch.Tensor,
        cond: torch.Tensor,
        tgt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.transform is None:
            return src, cond, tgt
        has_cond = cond is not None and cond.numel() > 0
        parts = [src, tgt] + ([cond] if has_cond else [])
        combined = torch.cat(parts, dim=0)
        out = self.transform.forward(combined)
        B = src.shape[0]
        if has_cond:
            src, tgt, cond = out.split([B, B, B], dim=0)
        else:
            src, tgt = out.split([B, B], dim=0)
        return src, cond, tgt

    @torch.no_grad()
    def get_test_data(
            self,
            device: torch.device,
            num_batches: int | None = None,
            return_metadata: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, Dict[str, list]]
    ):
        all_x0: list[torch.Tensor] = []
        all_x1: list[torch.Tensor] = []
        all_cond: list[torch.Tensor] = []

        dataset_size = self.dataset_size("test")
        total_batches = (dataset_size + self.batch_size - 1) // self.batch_size

        if num_batches is not None:
            total_batches = min(total_batches, num_batches)

        if total_batches <= 0:
            x0_empty, cond_empty, x1_empty = self.sample_at(indices=[], split="test")
            x0_empty = x0_empty.to(device)
            x1_empty = x1_empty.to(device)
            cond_empty = cond_empty.to(device) if cond_empty is not None else None
            print(f"  Pre-sampled {x0_empty.shape[0]} samples with shape {x0_empty.shape}")
            if return_metadata:
                return x0_empty, x1_empty, cond_empty, {}
            return x0_empty, x1_empty, cond_empty

        for _ in range(total_batches):
            x0, cond, x1 = self.sample(split="test")
            all_x0.append(x0.to(device))
            all_x1.append(x1.to(device))
            if cond is not None:
                all_cond.append(cond.to(device))

        x0_all = torch.cat(all_x0, dim=0)
        x1_all = torch.cat(all_x1, dim=0)
        cond_all = torch.cat(all_cond, dim=0) if all_cond else None
        print(f"  Pre-sampled {x0_all.shape[0]} samples with shape {x0_all.shape}")

        if return_metadata:
            dataset = self.datasets["test"]
            n_collected = x0_all.shape[0]
            if hasattr(dataset, "get_metadata_at"):
                metadata = dataset.get_metadata_at(list(range(n_collected)))
            else:
                metadata = {}
            return x0_all, x1_all, cond_all, metadata

        return x0_all, x1_all, cond_all

    def dataset_size(
        self,
        split: str
    ) -> int:
        self._validate_split(split)
        return len(self.datasets[split])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()

    def shutdown(self):
        if self._is_shutdown:
            return
        self._cleanup_all(list(self._loaders.values()))
        for k in self._iters:
            self._iters[k] = None
        self._is_shutdown = True
        if hasattr(self, "_finalizer") and self._finalizer.alive:
            self._finalizer.detach()

    @staticmethod
    def _cleanup(loader):
        try:
            it = getattr(loader, "_iterator", None)
            if it is not None and hasattr(it, "_shutdown_workers"):
                it._shutdown_workers()
        except Exception:
            pass

    @staticmethod
    def _cleanup_all(loaders):
        for loader in loaders:
            _BaseDatasetSampler._cleanup(loader)


class TSDatasetSampler(_BaseDatasetSampler):
    def __init__(
        self,
        root_dir                : str,
        source_spec             : Dict[str, Dict[str, List[str]]],
        target_spec             : Dict[str, List[str]],
        split                   : str              = "train",
        window_size             : int              = 2048,
        stride                  : Optional[int]    = None,
        val_ratio               : float            = 0.2,
        test_ratio              : float            = 0.2,
        seed                    : int              = 10,
        fs                      : int              = 500,
        batch_size              : int              = 256,
        num_workers             : int              = 4,
        prefetch_factor         : Optional[int]    = 4,
        pin_memory              : bool             = True,
        transform               : Optional[Any]    = None,
    ):
        super().__init__()

        subject2sessions = scan_tree(root_dir)
        train_s, val_s, test_s = train_val_test_split(
            subject2sessions,
            val_ratio  = val_ratio,
            test_ratio = test_ratio,
            seed       = seed
        )

        self.fs = fs
        self.window_size = window_size
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.transform = transform

        effective_stride = stride if stride is not None else window_size // 2
        self.stride = effective_stride

        self.datasets: Dict[str, TSWindowDataset] = {
            "train": TSWindowDataset(
                session_paths = train_s,
                source_spec   = source_spec,
                target_spec   = target_spec,
                window_size   = window_size,
                step          = effective_stride,
            ),
            "val": TSWindowDataset(
                session_paths = val_s,
                source_spec   = source_spec,
                target_spec   = target_spec,
                window_size   = window_size,
                step          = effective_stride,
            ),
            "test": TSWindowDataset(
                session_paths = test_s,
                source_spec   = source_spec,
                target_spec   = target_spec,
                window_size   = window_size,
                step          = effective_stride,
            ),
        }
        self._initialize_runtime(
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
        )

    def _make_empty_cond(
        self,
        *,
        batch_len: int,
        dataset: TSWindowDataset,
        ref_sample: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        window_size = getattr(dataset, "window_size", self.window_size)
        dtype = ref_sample['s'].dtype if ref_sample is not None else self.dummy.dtype
        return torch.zeros((batch_len, 0, window_size),
                           device=self.dummy.device,
                           dtype=dtype)


class BeatDatasetSampler(_BaseDatasetSampler):
    def __init__(
        self,
        root_dir                : str,
        source_spec             : Dict[str, Dict[str, List[str]]],
        target_spec             : Dict[str, List[str]],
        split                   : str              = "train",
        val_ratio               : float            = 0.2,
        test_ratio              : float            = 0.2,
        seed                    : int              = 10,
        batch_size              : int              = 256,
        num_workers             : int              = 4,
        prefetch_factor         : Optional[int]    = 4,
        pin_memory              : bool             = True,
        transform               : Optional[Any]    = None,
    ):
        super().__init__()

        subject2sessions = scan_tree(root_dir)
        train_s, val_s, test_s = train_val_test_split(
            subject2sessions,
            val_ratio  = val_ratio,
            test_ratio = test_ratio,
            seed       = seed
        )

        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.transform = transform

        self.datasets: Dict[str, BeatWindowDataset] = {
            "train": BeatWindowDataset(
                session_paths = train_s,
                source_spec   = source_spec,
                target_spec   = target_spec,
            ),
            "val": BeatWindowDataset(
                session_paths = val_s,
                source_spec   = source_spec,
                target_spec   = target_spec,
            ),
            "test": BeatWindowDataset(
                session_paths = test_s,
                source_spec   = source_spec,
                target_spec   = target_spec,
            ),
        }
        self._initialize_runtime(
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
        )

    def _make_empty_cond(
        self,
        *,
        batch_len: int,
        dataset: BeatWindowDataset,
        ref_sample: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if ref_sample is not None:
            beat_len = ref_sample['s'].shape[1]
            dtype = ref_sample['s'].dtype
        else:
            beat_len = 350
            dtype = self.dummy.dtype
        return torch.zeros((batch_len, 0, beat_len),
                           device=self.dummy.device,
                           dtype=dtype)

