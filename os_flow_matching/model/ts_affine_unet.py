import torch
import torch.nn as nn
from diffusers import UNet1DModel
from .vector_field import ConditionalVectorField


class TimeSeriesUNet(ConditionalVectorField):
    @staticmethod
    def _ensure_bcw(x: torch.Tensor, in_channels: int) -> torch.Tensor:
        if x.ndim == 2:
            if in_channels != 1:
                raise ValueError(f"Got (B, W) but in_channels={in_channels} != 1")
            return x.unsqueeze(1)  # (B, 1, W)
        if x.ndim == 3:
            B, A, Bdim = x.shape[0], x.shape[1], x.shape[2]
            if A == in_channels:        # (B, C, W)
                return x
            if Bdim == in_channels:     # (B, W, C) -> (B, C, W)
                return x.transpose(1, 2)
        raise ValueError(f"Expected (B, W), (B, C, W), or (B, W, C); got {tuple(x.shape)} with in_channels={in_channels}")

    @staticmethod
    def _ensure_bcw_any(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return x.unsqueeze(1)  # (B, 1, W)
        if x.ndim == 3:
            if x.shape[1] <= x.shape[2]:
                return x  # (B, C, W)
            return x.transpose(1, 2)  # (B, W, C) -> (B, C, W)
        raise ValueError(f"Expected (B, W), (B, C, W), or (B, W, C); got {tuple(x.shape)}")

    @staticmethod
    def _prepare_t(t: torch.Tensor, batch: int, device, dtype) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(batch)
        elif t.ndim == 1:
            if t.shape[0] == 1:
                t = t.expand(batch)
            elif t.shape[0] != batch:
                raise ValueError(f"t length {t.shape[0]} != batch {batch}")
        else:
            if t.shape[0] != batch:
                raise ValueError(f"t batch {t.shape[0]} != {batch}")
            t = t.view(batch, -1)
            if t.shape[1] != 1:
                raise ValueError("t must be scalar per sample (shape (B,) or (B,1,...))")
            t = t[:, 0]
        return t.to(device=device, dtype=torch.float32)

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        block_out_channels=(32, 64, 64),
        down_block_types=("DownBlock1D",) * 3,
        up_block_types=("UpBlock1D",) * 3,
        mid_block_type="UNetMidBlock1D",
        time_embedding_type="fourier",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.dropout = nn.Dropout1d(p=dropout) if dropout > 0 else nn.Identity()
        self.unet = UNet1DModel(
            sample_size=None,
            in_channels=in_channels,
            out_channels=out_channels,
            block_out_channels=tuple(block_out_channels),
            down_block_types=tuple(down_block_types),
            up_block_types=tuple(up_block_types),
            mid_block_type=mid_block_type,
            time_embedding_type=time_embedding_type,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
        x0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        has_x0 = x0 is not None and x0.numel() > 0
        has_cond = cond is not None and cond.numel() > 0

        if has_x0 or has_cond:
            x_bcw = self._ensure_bcw_any(x)
            if has_x0:
                x0_bcw = self._ensure_bcw_any(x0)
                x_bcw = torch.cat([x_bcw, x0_bcw], dim=1)
            if has_cond:
                cond_bcw = self._ensure_bcw_any(cond)
                x_bcw = torch.cat([x_bcw, cond_bcw], dim=1)
        else:
            x_bcw = self._ensure_bcw(x, self.in_channels)

        if x_bcw.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected in_channels={self.in_channels}, got {x_bcw.shape[1]} after concatenation"
            )

        t_flat = self._prepare_t(t, x_bcw.shape[0], x_bcw.device, x_bcw.dtype)

        t_flat = 1.0 - t_flat

        x_bcw = self.dropout(x_bcw)
        return self.unet(sample=x_bcw, timestep=t_flat, return_dict=True).sample
