import torch
import torch.nn as nn
from diffusers import UNet1DModel
from .vector_field import ConditionalVectorField


class CylindricalGeodesicUNet(ConditionalVectorField):
    """
    UNet for cylindrical Hilbert representation.
    Input: (B, W, 3) or (B, 3, W) where channels are [cos_phi, sin_phi, amp]
    Output: same format
    """
    @staticmethod
    def _ensure_bcw(x: torch.Tensor, in_channels: int = 3) -> torch.Tensor:
        if x.ndim == 3:
            B, A, Bdim = x.shape[0], x.shape[1], x.shape[2]
            if A == in_channels:        # (B, C, W)
                return x
            if Bdim == in_channels:     # (B, W, C) -> (B, C, W)
                return x.transpose(1, 2)
        if x.ndim == 4:
            # Handle (B, 1, W, C) by squeezing singleton dimension
            if x.shape[1] == 1:
                x = x.squeeze(1)  # (B, W, C)
                if x.shape[2] == in_channels:
                    return x.transpose(1, 2)  # (B, C, W)
        raise ValueError(f"Expected (B, C, W) or (B, W, C) with C={in_channels}; got {tuple(x.shape)}")

    @staticmethod
    def _infer_in_channels_real(x: torch.Tensor) -> int:
        if x.ndim == 2:
            return 1
        if x.ndim == 3:
            return x.shape[1] if x.shape[1] <= x.shape[2] else x.shape[2]
        if x.ndim == 4 and x.shape[1] == 1:
            return x.shape[3]
        raise ValueError(f"Cannot infer channel count from shape {tuple(x.shape)}")

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
        in_channels: int = 3,
        out_channels: int = 2,
        block_out_channels=(32, 32, 64, 64),
        down_block_types=("DownBlock1D",) * 4,
        up_block_types=("UpBlock1D",) * 4,
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
        original_shape = x.shape
        x_channels = self._infer_in_channels_real(x)
        x_bcw = self._ensure_bcw(x, x_channels)
        x_was_b1wc = len(original_shape) == 4 and original_shape[1] == 1
        x_was_bwc = len(original_shape) == 3 and original_shape[2] == x_channels
        t_flat = self._prepare_t(t, x_bcw.shape[0], x_bcw.device, x_bcw.dtype)

        x_input = x_bcw
        if x0 is not None and x0.numel() > 0:
            x0_channels = self._infer_in_channels_real(x0)
            x0_bcw = self._ensure_bcw(x0, x0_channels)
            x_input = torch.cat([x_input, x0_bcw], dim=1)

        if cond is not None and cond.numel() > 0:
            cond_channels = self._infer_in_channels_real(cond)
            cond_bcw = self._ensure_bcw(cond, cond_channels)
            x_input = torch.cat([x_input, cond_bcw], dim=1)

        if x_input.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected in_channels={self.in_channels}, got {x_input.shape[1]} after concatenation"
            )
        
        x_input = self.dropout(x_input)
        # Predict intrinsic parameters: [angular_velocity, amp_velocity]
        intrinsic_pred = self.unet(sample=x_input, timestep=t_flat, return_dict=True).sample

        # Decompose input geometry
        cos_phi = x_bcw[:, 0:1, :]
        sin_phi = x_bcw[:, 1:2, :]

        omega = intrinsic_pred[:, 0:1, :]  
        du = intrinsic_pred[:, 1:2, :]

        v_cos = -sin_phi * omega
        v_sin = cos_phi * omega
        v_amp = du

        output = torch.cat([v_cos, v_sin, v_amp], dim=1)

        if x_was_b1wc:
            output = output.transpose(1, 2).unsqueeze(1)
        elif x_was_bwc:
            output = output.transpose(1, 2)

        return output
