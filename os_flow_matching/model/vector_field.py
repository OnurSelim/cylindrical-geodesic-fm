from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn


class ConditionalVectorField(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,   # (bs, c_x, w)
        t: torch.Tensor,   # (bs, 1, 1)
        cond: Optional[torch.Tensor] = None,
        x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:     # (bs, c_x, w)
        raise NotImplementedError