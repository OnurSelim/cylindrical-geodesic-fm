from __future__ import annotations
import math

import torch
import torch.nn.functional as F
from torch import Tensor

from .hilbert import unwrap, hilbert


def _moving_average(y: torch.Tensor, span: int) -> torch.Tensor:
    span = max(5, int(span) | 1)
    if span >= y.shape[-1]:
        return y.clone()

    kernel = torch.ones(span, dtype=y.dtype, device=y.device) / float(span)
    pad    = span // 2
    y_pad  = F.pad(y.reshape(-1, 1, y.shape[-1]), (pad, pad),
                   mode='constant', value=0.0)
    out    = F.conv1d(y_pad, kernel.view(1, 1, span)).view_as(y)
    return out


def _mad_outlier_mask(y: torch.Tensor, thresh: float = 3.5) -> torch.Tensor:
    med = torch.median(y, dim=-1, keepdim=True).values
    mad = torch.median((y - med).abs(), dim=-1, keepdim=True).values
    z   = torch.zeros_like(y)
    z   = torch.where(mad != 0,
                      0.6745 * (y - med) / mad,
                      z)
    return torch.abs(z) > thresh


def _linear_interp_bad(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return y.clone()
    B = y.numel() // y.shape[-1]
    T = y.shape[-1]
    y_flat    = y.reshape(B, T)
    mask_flat = mask.reshape(B, T)
    device    = y.device
    dtype     = y.dtype

    idx = torch.arange(T, device=device)
    idx = idx.expand(B, T)

    valid          = ~mask_flat
    prev_idx_val   = torch.where(valid, idx, torch.zeros_like(idx))
    prev_idx       = torch.cummax(prev_idx_val, dim=-1)[0]

    next_idx_val   = torch.where(valid, idx, torch.full_like(idx, T - 1))
    next_idx       = torch.flip(next_idx_val, dims=[-1])
    next_idx       = T - 1 - torch.cummin(next_idx, dim=-1)[0]
    next_idx       = torch.flip(next_idx, dims=[-1])

    y_prev = y_flat.gather(-1, prev_idx)
    y_next = y_flat.gather(-1, next_idx)

    denom  = (next_idx - prev_idx).clamp(min=1).to(dtype)
    w      = (idx.to(dtype) - prev_idx.to(dtype)) / denom
    y_interp = y_prev + w * (y_next - y_prev)

    out = torch.where(mask_flat, y_interp, y_flat)
    return out.view_as(y)


def _build_full_spectrum_from_half(c_half: torch.Tensor) -> torch.Tensor:
    N2 = c_half.shape[-1] - 1
    N  = 2 * N2
    c  = torch.empty(c_half.shape[:-1] + (N,),
                     dtype=c_half.dtype, device=c_half.device)

    c[..., :N2 + 1] = c_half
    if N2 > 1:
        c[..., N2 + 1:] = torch.conj(c_half[..., 1:N2].flip(-1))
    else:
        c[..., N2 + 1:] = torch.empty(c_half.shape[:-1] + (0,),
                                      dtype=c_half.dtype,
                                      device=c_half.device)
    return c


def _correct_piecewise(cH: torch.Tensor, m: torch.Tensor | int) -> torch.Tensor:
    if not torch.is_complex(cH):
        raise TypeError("cH must be a complex tensor")

    *batch_shape, L = cH.shape
    N2 = L - 1
    device = cH.device

    m = torch.as_tensor(m, device=device, dtype=torch.long)
    if m.dim() == 0:
        if batch_shape:
            m = m.expand(batch_shape)
    else:
        m = m.expand(batch_shape)

    if batch_shape:
        m = m.reshape(*batch_shape)
    else:
        m = m.reshape(())

    idx = torch.arange(L, device=device, dtype=torch.long)
    idx = idx.view((1,) * len(batch_shape) + (L,))
    m_e = m.unsqueeze(-1)

    def _get(spec: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        k_reflect = (k > N2) | (k < 0)
        k_eff = torch.where(k_reflect, 2 * N2 - k, k)
        out   = torch.take_along_dim(spec, k_eff.clamp(0, N2), -1)
        return torch.where(k_reflect, out.conj(), out)

    start = torch.clamp(N2 - 2 * m + 1, min=0)
    start_e = start.unsqueeze(-1)

    cP = torch.where(idx >= start_e, 2.0 * cH, torch.zeros_like(cH))

    n_passes = torch.ceil((N2 - 2 * m + 1).to(torch.float32) / (2 * m))
    n_passes = n_passes.clamp(min=0).to(torch.int64)
    passes = int(n_passes.max().item())

    for _ in range(passes):
        upper = torch.clamp(N2 - 2 * m, min=-1)
        # Paper's implementation
        # mask_hi = (idx >= (m_e + 1)) & (idx <= upper.unsqueeze(-1))
        # rhs1 = 2.0 * cH - 0.5 * _get(cP, idx + 2 * m_e).conj()
        # cP   = torch.where(mask_hi, rhs1, cP)

        # Corrected version
        mask_hi = (idx >= (m_e + 1)) & (idx <= upper.unsqueeze(-1))
        rhs1 = 2.0 * cH + _get(cP, idx + 2 * m_e)
        cP = torch.where(mask_hi, rhs1, cP)

        k_m  = m_e
        k_3m = 3 * m_e
        c3m  = _get(cP, k_3m).squeeze(-1)

        cH_m = torch.take_along_dim(cH, k_m, -1).squeeze(-1)
        real = (2.0 * cH_m + c3m).real
        imag = (cH_m + 0.5 * c3m).imag
        cP_m = torch.complex(real, imag)

        cP = cP.clone()
        cP.scatter_(-1, k_m, cP_m.unsqueeze(-1))

        mask_lo = idx < m_e
        rhs2 = (cH
                + 0.5 * _get(cP, 2 * m_e - idx).conj()
                + 0.5 * _get(cP, idx + 2 * m_e))
        cP = torch.where(mask_lo, rhs2, cP)
    return cP


def _clamp_phase_velocity(phi: torch.Tensor, max_dphi: float) -> torch.Tensor:
    if max_dphi <= 0:
        return phi
    dphi = torch.diff(phi, dim=-1)
    dphi = dphi.clamp(min=-max_dphi, max=max_dphi)
    phi0 = phi[..., :1]
    return torch.cat([phi0, phi0 + torch.cumsum(dphi, dim=-1)], dim=-1)


def eht_transform(
    y: torch.Tensor,
    *,
    fs: float | None = None,
    dt: float  | None = None,
    smooth_frac: float = 0.02,
    mad_thresh: float  = 3.5,
    eps: float = 1e-12,
) -> tuple[Tensor, Tensor, Tensor]:

    if y.ndim < 1:
        raise ValueError("extended_hilbert_phase expects at least 1-D input")
    if dt is None:
        if fs is None:
            raise ValueError("Provide either fs or dt")
        dt = 1.0 / float(fs)
    else:
        dt = float(dt)

    y = y.contiguous()
    device, dtype = y.device, y.dtype
    T             = y.shape[-1]
    time_axis     = -1

    z      = hilbert(y)
    amp = z.abs().clamp_min(eps)

    phi_H  = unwrap(torch.angle(z), dim=time_axis)

    omega_hat = (phi_H[..., -1] - phi_H[..., 0]) / ((T - 1) * dt)
    omega_hat_expand = omega_hat.unsqueeze(-1)

    t_vec = (torch.arange(T, device=device, dtype=dtype) * dt).view(
            (1,) * (y.dim() - 1) + (T,))

    u_H = phi_H - omega_hat_expand * t_vec

    if T & 1:
        u_work = u_H[..., :-1]
    else:
        u_work = u_H
    N  = u_work.shape[-1]

    cH       = torch.fft.fft(u_work, dim=time_axis)
    cH_half  = cH[..., : N // 2 + 1].clone()

    m_raw = omega_hat * N * dt / (2.0 * math.pi)
    m_int = torch.clamp(m_raw.round().to(torch.int64), 1, N // 2 - 1)

    cP_half = _correct_piecewise(cH_half, m_int)
    u_hat_work = torch.fft.ifft(_build_full_spectrum_from_half(cP_half),
                                dim=time_axis).real

    if T != N:
        slope  = u_hat_work[..., -1:] - u_hat_work[..., -2:-1]
        u_hat  = torch.cat([u_hat_work, u_hat_work[..., -1:] + slope], dim=time_axis)
    else:
        u_hat  = u_hat_work

    span      = max(5, int(math.floor(smooth_frac * T)))
    u_smooth  = _moving_average(u_hat, span)
    mask      = _mad_outlier_mask(u_smooth, mad_thresh)
    u_smooth  = _linear_interp_bad(u_smooth, mask)
    u_smooth = _clamp_phase_velocity(u_smooth, float(0.45))
    phi_hat = unwrap(omega_hat_expand * t_vec + u_smooth, dim=time_axis)

    return amp, phi_hat, omega_hat


def eht_inverse(amp: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    if amp.shape != phase.shape:
        raise ValueError("amp and phase must have the same shape")
    return amp * torch.cos(phase)


