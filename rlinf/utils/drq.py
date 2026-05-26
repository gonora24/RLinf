# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F


@torch.no_grad()
def crop_bchw_fast(x, pad=4):
    B, C, H, W = x.shape
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")

    Hp, Wp = x.shape[-2], x.shape[-1]
    max_top = Hp - H
    max_left = Wp - W

    top = torch.randint(0, max_top + 1, (B,), device=x.device)
    left = torch.randint(0, max_left + 1, (B,), device=x.device)

    windows = x.unfold(2, H, 1).unfold(3, W, 1)  # [B,C,Hp-H+1,Wp-W+1,H,W]
    b = torch.arange(B, device=x.device)
    out = windows[b, :, top, left, :, :]
    return out.contiguous()


@torch.no_grad()
def drq_crop_main(x, pad=4):
    # accept BCHW or BHWC
    if x.ndim != 4:
        raise ValueError(f"main_images expected 4D, got {tuple(x.shape)}")

    if x.shape[1] == 3:  # BCHW
        return crop_bchw_fast(x, pad=pad)

    if x.shape[-1] == 3:  # BHWC -> BCHW -> crop -> BHWC
        x_bchw = x.permute(0, 3, 1, 2).contiguous()
        y_bchw = crop_bchw_fast(x_bchw, pad=pad)
        return y_bchw.permute(0, 2, 3, 1).contiguous()

    raise ValueError(
        f"main_images expected [B,3,H,W] or [B,H,W,3], got {tuple(x.shape)}"
    )


@torch.no_grad()
def drq_crop_extra(x, pad=4):
    if x is None:
        return None
    if x.ndim != 5:
        raise ValueError(f"extra_view_images expected 5D, got {tuple(x.shape)}")

    # accept BVCHW or BVHWC
    if x.shape[2] == 3:  # [B,V,3,H,W]
        B, V, C, H, W = x.shape
        y = x.reshape(B * V, C, H, W).contiguous()
        y = crop_bchw_fast(y, pad=pad)
        return y.view(B, V, C, H, W).contiguous()

    if x.shape[-1] == 3:  # [B,V,H,W,3]
        B, V, H, W, C = x.shape
        y = x.permute(0, 1, 4, 2, 3).contiguous().view(B * V, C, H, W)
        y = crop_bchw_fast(y, pad=pad)
        y = y.view(B, V, C, H, W).permute(0, 1, 3, 4, 2).contiguous()
        return y

    raise ValueError(
        f"extra_view_images expected C=3 in dim2 or last dim, got {tuple(x.shape)}"
    )


@torch.no_grad()
def resize_main_images_for_dsrl(obs, size=64):
    """Resize main camera images to DSRL critic resolution before augmentation.

    Replay buffers store env-resolution images (e.g. 256x256). DRQ and color
    jitter should run at the same 64x64 resolution used by dsrl_pi0 and the
    DSRL encoder, not at full resolution.
    """
    if obs.get("main_images") is None:
        return obs

    x = obs["main_images"]
    if x.ndim != 4:
        raise ValueError(f"main_images expected 4D, got {tuple(x.shape)}")

    if x.shape[1] == 3:  # BCHW
        x_bchw = x
        return_layout = "bchw"
    elif x.shape[-1] == 3:  # BHWC
        x_bchw = x.permute(0, 3, 1, 2).contiguous()
        return_layout = "bhwc"
    else:
        raise ValueError(
            f"main_images expected [B,3,H,W] or [B,H,W,3], got {tuple(x.shape)}"
        )

    _, _, h, w = x_bchw.shape
    if h == size and w == size:
        return obs

    if x_bchw.dtype == torch.uint8:
        x_float = x_bchw.float() / 255.0
    else:
        x_float = x_bchw.float()
        if x_float.min() < 0:
            x_float = (x_float + 1.0) / 2.0

    resized = F.interpolate(
        x_float, size=(size, size), mode="bilinear", align_corners=False
    )
    if x_bchw.dtype == torch.uint8:
        obs["main_images"] = (resized.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        if return_layout == "bhwc":
            obs["main_images"] = obs["main_images"].permute(0, 2, 3, 1).contiguous()
    else:
        resized = resized.to(dtype=x_bchw.dtype)
        obs["main_images"] = (
            resized.permute(0, 2, 3, 1).contiguous()
            if return_layout == "bhwc"
            else resized
        )

    return obs


@torch.no_grad()
def apply_drq(obs, pad=4):
    """
    Apply DRQ (random crop data regularization) to an observation dict.

    DRQ / DRP is commonly used in pixel-based RL algorithms (e.g., DrQ, DrQ-v2)
    to stabilize training and improve generalization by augmenting visual inputs.

    This function modifies the observation dict in-place.

    Expected keys:
      - "main_images": main camera images
      - "extra_view_images": optional multi-view images

    Args:
        obs (dict): Observation dictionary.
        pad (int): Padding size for random crop.

    Returns:
        dict: Observation dictionary with DRQ applied.
    """
    if 'main_images' in obs.keys() and obs["main_images"] is not None:
    # if obs.get("main_images", None) is not None:
        obs["main_images"] = drq_crop_main(obs["main_images"], pad=pad)

    if 'extra_view_images' in obs.keys() and obs["extra_view_images"] is not None:
        obs["extra_view_images"] = drq_crop_extra(obs["extra_view_images"], pad=pad)

    return obs
