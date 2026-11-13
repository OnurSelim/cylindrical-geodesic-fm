import torch
from torch import nn, Tensor
from typing import Optional
from abc import ABC, abstractmethod

from .dataset.transform import Transform
from .dataset.sampleable import Sampleable


class ConditionalProbabilityPath(nn.Module, ABC):
    def __init__(self, sampler: Sampleable):
        super().__init__()
        self.sampler = sampler

    @property
    def transform(self) -> Transform:
        return self.sampler.transform

    def to_time_series(self, x: Tensor) -> Tensor:
        return self.transform.backward(x)

    def to_path_coords(self, x: Tensor) -> Tensor:
        return x

    def to_simulation_coords(self, x: Tensor) -> Tensor:
        return x

    def from_simulation_coords(self, x: Tensor) -> Tensor:
        return x

    def project_simulation_coords(self, x: Tensor) -> Tensor:
        return x

    @staticmethod
    def _expand_t_like(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        while t.ndim < like.ndim:
            t = t.unsqueeze(-1)
        return t

    @abstractmethod
    def _interpolate(
        self,
        x0: Tensor,
        x1: Tensor,
        t: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError

    def compute_conditional_path(
        self,
        x0: Tensor,
        x1: Tensor,
        cond: Optional[Tensor] = None,
        *,
        t: Optional[Tensor] = None,
        already_transformed: bool = False,
    ) -> tuple[tuple[Tensor, Tensor, Tensor], Tensor, Optional[Tensor]]:
        if not already_transformed:
            x0 = self.transform.forward(x0)
            x1 = self.transform.forward(x1)
            if cond is not None and cond.numel() > 0:
                cond = self.transform.forward(cond)

        if t is None:
            bs = x0.shape[0]
            t = torch.rand(bs, 1, device=x0.device, dtype=x0.dtype)
        else:
            if t.device != x0.device or t.dtype != x0.dtype:
                t = t.to(device=x0.device, dtype=x0.dtype)

        # Interpolate and return in transform space.
        # Callers convert to path coords where needed (model, simulator).
        xt, x0t, x1t = self._interpolate(x0, x1, t)
        return (xt, x0t, x1t), t, cond

    def sample_conditional_path(
        self,
        split: str = "train"
    ) -> tuple[tuple[Tensor, Tensor, Tensor], Tensor, Optional[Tensor]]:
        x0, cond, x1 = self.sampler.sample(split=split)
        return self.compute_conditional_path(
            x0=x0,
            x1=x1,
            cond=cond,
            already_transformed=True,
        )

    @abstractmethod
    def conditional_vector_field(
        self,
        x0: Tensor,
        t:  Tensor,
        *,
        xt: Optional[Tensor] = None,
        x1: Optional[Tensor] = None,
        cond: Optional[Tensor] = None,
    ) -> Tensor:
        raise NotImplementedError

class TimeSeriesAffinePath(ConditionalProbabilityPath):
    def _interpolate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        t = self._expand_t_like(t, x0)
        return (1.0 - t) * x0 + t * x1, x0, x1

    def conditional_vector_field(
        self,
        x0: Tensor,
        t:  Tensor,
        *,
        xt: Optional[Tensor] = None,
        x1: Optional[Tensor] = None,
        cond: Optional[Tensor] = None,
    ) -> Tensor:
        v = x1 - x0
        return v

class HilbertNaivePath(ConditionalProbabilityPath):
    def __init__(self, sampler: Sampleable):
        super().__init__(sampler=sampler)

    def _interpolate(
        self,
        x0: Tensor,
        x1: Tensor,
        t: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if x0.shape != x1.shape:
            raise ValueError(
                f"x0 and x1 must have the same shape, got {x0.shape} vs {x1.shape}"
            )
        if x0.device != x1.device:
            raise ValueError("x0 and x1 must be on the same device")

        amp0, phi0 = x0.unbind(dim=-1)
        amp1, phi1 = x1.unbind(dim=-1)

        t_bc = self._expand_t_like(t.to(x0.device, x0.dtype), amp0)

        amp_t = (1.0 - t_bc) * amp0 + t_bc * amp1

        delta_phi = phi1 - phi0
        phi_t = phi0 + t_bc * delta_phi

        xt  = torch.stack([amp_t, phi_t], dim=-1)
        x0t = torch.stack([amp0,  phi0], dim=-1)
        x1t = torch.stack([amp1,  phi1], dim=-1)
        return xt, x0t, x1t

    def conditional_vector_field(
        self,
        x0: Tensor,
        t: Tensor,
        *,
        xt: Optional[Tensor] = None,
        x1: Optional[Tensor] = None,
        cond: Optional[Tensor] = None,
    ) -> Tensor:
        if x1 is None:
            raise ValueError("x1 must be provided to compute the conditional vector field.")

        amp0, phi0 = x0.unbind(dim=-1)
        amp1, phi1 = x1.unbind(dim=-1)

        delta_phi = phi1 - phi0

        d_amp = amp1 - amp0
        return torch.stack([d_amp, delta_phi], dim=-1)

class CylindricalGeodesicPath(ConditionalProbabilityPath):
    def __init__(self, sampler: Sampleable):
        super().__init__(sampler=sampler)

    def to_path_coords(self, x: Tensor) -> Tensor:
        amp, phi = x.unbind(dim=-1)
        return torch.stack([torch.cos(phi), torch.sin(phi), amp], dim=-1)

    def _interpolate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if x0.shape != x1.shape:
            raise ValueError(
                f"x0 and x1 must have the same shape, got {x0.shape} vs {x1.shape}"
            )
        if x0.device != x1.device:
            raise ValueError("x0 and x1 must be on the same device")

        amp0, phi0 = x0.unbind(dim=-1)
        amp1, phi1 = x1.unbind(dim=-1)

        t_bc = self._expand_t_like(t.to(x0.device, x0.dtype), amp0)

        amp_t = (1.0 - t_bc) * amp0 + t_bc * amp1
        phi_t = phi0 + t_bc * (phi1 - phi0)

        xt = torch.stack([amp_t, phi_t], dim=-1)
        return xt, x0, x1

    def conditional_vector_field(
            self,
            x0: Tensor,
            t: Tensor,
            *,
            xt: Optional[Tensor] = None,
            x1: Optional[Tensor] = None,
            cond: Optional[Tensor] = None,
    ) -> Tensor:
        if xt is None:
            raise ValueError("xt must be provided for the cylindrical vector field.")
        if x1 is None:
            raise ValueError("x1 must be provided for the cylindrical vector field.")

        # xt, x0, x1 are all in transform space (amp, phi)
        amp_t, phi_t = xt.unbind(dim=-1)
        amp0, phi0 = x0.unbind(dim=-1)
        amp1, phi1 = x1.unbind(dim=-1)

        cos_t = torch.cos(phi_t)
        sin_t = torch.sin(phi_t)

        dphi = phi1 - phi0
        r = amp1 - amp0

        vx = -sin_t * dphi
        vy = cos_t * dphi
        vz = r

        return torch.stack([vx, vy, vz], dim=-1)

    def project_simulation_coords(self, x: Tensor) -> Tensor:
        cos_phi, sin_phi, amp = x.unbind(dim=-1)
        radius = torch.sqrt(cos_phi.square() + sin_phi.square()).clamp_min(torch.finfo(x.dtype).eps)
        cos_phi = cos_phi / radius
        sin_phi = sin_phi / radius
        return torch.stack([cos_phi, sin_phi, amp], dim=-1)

    def to_time_series(self, x: Tensor) -> Tensor:
        cos_phi, sin_phi, amp = x.unbind(dim=-1)
        phi = torch.atan2(sin_phi, cos_phi)
        return self.transform.backward(torch.stack([amp, phi], dim=-1))

