from __future__ import annotations

import math
import torch
from torch import Tensor
from typing import Dict, Tuple


_H_CACHE: Dict[Tuple[int, torch.dtype, str, int], torch.Tensor] = {}


def _hilbert_filter(N: int,
                    dtype: torch.dtype,
                    device: torch.device) -> torch.Tensor:
    key = (N, dtype, device.type, device.index if device.type == "cuda" else -1)
    H = _H_CACHE.get(key, None)
    if H is not None and H.device == device and H.dtype == dtype:
        return H

    H = torch.zeros(N, dtype=dtype, device=device)
    if N % 2 == 0:
        H[0] = 1
        H[N // 2] = 1
        if N // 2 > 1:
            H[1:N // 2] = 2
    else:
        H[0] = 1
        H[1:(N + 1) // 2] = 2

    _H_CACHE[key] = H
    return H


def hilbert(x: torch.Tensor) -> torch.Tensor:
    if not x.is_floating_point():
        raise TypeError("hilbert expects a floating-point real tensor")

    N = x.shape[-1]
    Xf = torch.fft.fft(x, dim=-1)
    H = _hilbert_filter(N, Xf.dtype, x.device)
    Z = torch.fft.ifft(Xf * H, dim=-1)
    return Z


def unwrap(phi: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    if not phi.is_floating_point():
        raise TypeError("unwrap expects a real-valued float tensor")

    d = torch.diff(phi, dim=dim)
    pi     = math.pi
    twopi  = 2.0 * pi

    dd = torch.remainder(d + pi, twopi) - pi
    dd = torch.where((dd == -pi) & (d > 0), torch.full_like(dd, pi), dd)

    correction = dd - d
    correction = torch.where(torch.abs(correction) < 1e-12,
                             torch.zeros_like(correction),
                             correction)
    c = torch.cumsum(correction, dim=dim)

    pad_shape      = list(phi.shape)
    if dim < 0:
        dim = phi.dim() + dim
    pad_shape[dim] = 1
    c0 = torch.zeros(pad_shape, dtype=phi.dtype, device=phi.device)
    return phi + torch.cat([c0, c], dim=dim)


def ht_transform(
    y: torch.Tensor,
    *,
    fs: float | None = None,
    dt: float | None = None,
    eps: float = 1e-12,
) -> Tuple[Tensor, Tensor, Tensor]:
    if y.ndim < 1:
        raise ValueError("ht_transform expects at least 1-D input")
    if dt is None:
        if fs is None:
            raise ValueError("Provide either fs or dt")
        dt = 1.0 / float(fs)
    else:
        dt = float(dt)
    
    y = y.contiguous()
    T = y.shape[-1]
    time_axis = -1
    
    # Compute analytic signal via Hilbert transform
    z = hilbert(y)
    
    # Extract amplitude (instantaneous envelope)
    amp = z.abs().clamp_min(eps)
    
    # Extract and unwrap phase
    phi = unwrap(torch.angle(z), dim=time_axis)
    
    # Estimate mean angular frequency
    omega_hat = (phi[..., -1] - phi[..., 0]) / ((T - 1) * dt)
    
    return amp, phi, omega_hat


def ht_inverse(amp: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    if amp.shape != phase.shape:
        raise ValueError("amp and phase must have the same shape")
    return amp * torch.cos(phase)


