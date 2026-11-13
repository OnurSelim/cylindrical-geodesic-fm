from __future__ import annotations

import torch
from torch import Tensor
from typing import Optional
from abc import ABC, abstractmethod

from settings import FS


class Transform(ABC):
    def __init__(self, model_window_size: Optional[int] = None):
        self._model_window_size = model_window_size

    @property
    def model_window_size(self) -> Optional[int]:
        return self._model_window_size

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def backward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def crop_to_model_size(self, x: Tensor) -> Tensor:
        from ..utils.general import crop_center
        ws = self.model_window_size
        if ws is not None and ws < x.shape[-1]:
            return crop_center(x, ws)
        return x

    def to_time_series(self, x: Tensor) -> Tensor:
        return self.backward(x)


class Identity(Transform):
    def forward(self, x: Tensor) -> Tensor:
        return self.crop_to_model_size(x)

    def backward(self, x: Tensor) -> Tensor:
        return x
    

class Hilbert(Transform):
    def forward(self, x: Tensor) -> Tensor:
        from ..utils.hilbert import ht_transform

        amp, phi, _ = ht_transform(x, fs=FS)
        amp = self.crop_to_model_size(amp)
        phi = self.crop_to_model_size(phi)
        return torch.stack([amp, phi], dim=-1)

    def backward(self, x: Tensor) -> Tensor:
        from ..utils.hilbert import ht_inverse
        
        amp, phi = x.unbind(dim=-1)
        return ht_inverse(amp, phi)


class ExtendedHilbert(Transform):
    def forward(self, x: Tensor) -> Tensor:
        from ..utils.hilbert_ext import eht_transform

        amp, phi, _ = eht_transform(x, fs=FS)
        amp = self.crop_to_model_size(amp)
        phi = self.crop_to_model_size(phi)
        return torch.stack([amp, phi], dim=-1)

    def backward(self, x: Tensor) -> Tensor:
        from ..utils.hilbert_ext import eht_inverse

        amp, phi = x.unbind(dim=-1)
        return eht_inverse(amp, phi)


TRANSFORMS = {
    "identity": Identity,
    "hilbert": Hilbert,
    "extended_hilbert": ExtendedHilbert,
}


