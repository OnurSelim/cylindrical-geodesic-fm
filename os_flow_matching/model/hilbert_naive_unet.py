import torch
import torch.nn as nn
from diffusers import UNet1DModel
from .vector_field import ConditionalVectorField


class HilbertNaiveUNet(ConditionalVectorField):
    """
    UNet-based conditional vector field for HilbertNaivePath.

    Expected input:
        x: (B, C, L, 2)   where last dim = [amp, phi]

    Internal UNet representation:
        x_unet: (B, 2*C, L)

    Output:
        v_hat: (B, C, L, 2)   predicted velocity in (a, phi) coordinates.
    """
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
        in_channels: int = 2,
        out_channels: int = 2,
        block_out_channels=(32, 32, 64),
        down_block_types=("DownBlock1D",) * 3,
        up_block_types=("UpBlock1D",) * 3,
        mid_block_type="UNetMidBlock1D",
        time_embedding_type="fourier",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if in_channels % 2 != 0:
            raise ValueError(
                f"in_channels must be even (2 * C for amp/phi), got {in_channels}"
            )
        if out_channels % 2 != 0:
            raise ValueError(
                f"out_channels must be even (2 * C for amp/phi), got {out_channels}"
            )

        self.num_input_signal_channels = in_channels // 2
        self.num_output_signal_channels = out_channels // 2
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
        if x.ndim != 4 or x.shape[-1] != 2:
            raise ValueError(
                f"Expected x of shape (B, C, L, 2) with last dim [amp, phi], "
                f"got {tuple(x.shape)}"
            )

        source_channels = x.shape[1]
        if source_channels != self.num_output_signal_channels:
            raise ValueError(
                f"Input source channels C={source_channels} do not match "
                f"model output channels C={self.num_output_signal_channels}"
            )

        if x0 is not None and x0.numel() > 0:
            if x0.ndim != 4 or x0.shape[-1] != 2:
                raise ValueError(
                    f"Expected x0 of shape (B, C0, L, 2) with last dim [amp, phi], "
                    f"got {tuple(x0.shape)}"
                )
            if x0.shape[0] != x.shape[0] or x0.shape[2] != x.shape[2]:
                raise ValueError(
                    f"x0 shape {tuple(x0.shape)} incompatible with x {tuple(x.shape)}"
                )
            x = torch.cat([x, x0], dim=1)

        if cond is not None and cond.numel() > 0:
            if cond.ndim != 4 or cond.shape[-1] != 2:
                raise ValueError(
                    f"Expected cond of shape (B, Cc, L, 2) with last dim [amp, phi], "
                    f"got {tuple(cond.shape)}"
                )
            if cond.shape[0] != x.shape[0] or cond.shape[2] != x.shape[2]:
                raise ValueError(
                    f"cond shape {tuple(cond.shape)} incompatible with x {tuple(x.shape)}"
                )
            x = torch.cat([x, cond], dim=1)

        B, C, L, two = x.shape
        if C != self.num_input_signal_channels:
            raise ValueError(
                f"Input C={C} does not match model.num_input_signal_channels={self.num_input_signal_channels}"
            )

        x_bcw = x.permute(0, 3, 1, 2).reshape(B, 2 * C, L)
        t_flat = self._prepare_t(t, B, x_bcw.device, x_bcw.dtype)
        t_flat = 1.0 - t_flat
        x_bcw = self.dropout(x_bcw)
        y = self.unet(sample=x_bcw, timestep=t_flat, return_dict=True).sample
        y = y.view(B, 2, self.num_output_signal_channels, L).permute(0, 2, 3, 1).contiguous()

        return y