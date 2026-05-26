"""
Verifizierte PyTorch-Implementierung der JAX Color-Augmentation via torch.func.vmap.
Bietet maximale GPU-Performance durch native Vektorisierung ohne Python-Batch-Schleifen.
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import torch
from torch.func import vmap  # Erfordert PyTorch >= 2.0


# ---------------------------------------------------------------------------
# Farbkonvertierungen (Pure Functions)
# ---------------------------------------------------------------------------

def rgb_to_hsv(
    r: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vv     = torch.maximum(torch.maximum(r, g), b)
    range_ = vv - torch.minimum(torch.minimum(r, g), b)

    sat  = torch.where(vv > 0, range_ / vv, torch.zeros_like(vv))
    norm = torch.where(
        range_ != 0,
        1.0 / (6.0 * range_),
        torch.full_like(range_, 1e9),
    )

    hr = norm * (g - b)
    hg = norm * (b - r) + 2.0 / 6.0
    hb = norm * (r - g) + 4.0 / 6.0

    hue = torch.where(r == vv, hr, torch.where(g == vv, hg, hb))
    hue = hue * (range_ > 0).to(hue.dtype)
    hue = hue + (hue < 0).to(hue.dtype)

    return hue, sat, vv


def hsv_to_rgb(
    h: torch.Tensor,
    s: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    c    = s * v
    m    = v - c
    dh   = (h % 1.0) * 6.0
    fmod = dh % 2.0
    x    = c * (1.0 - torch.abs(fmod - 1.0))
    cat  = torch.floor(dh).long()

    zeros = torch.zeros_like(c)
    rr = torch.where((cat == 0) | (cat == 5), c,
         torch.where((cat == 1) | (cat == 4), x, zeros)) + m
    gg = torch.where((cat == 1) | (cat == 2), c,
         torch.where((cat == 0) | (cat == 3), x, zeros)) + m
    bb = torch.where((cat == 3) | (cat == 4), c,
         torch.where((cat == 2) | (cat == 5), x, zeros)) + m

    return rr, gg, bb


# ---------------------------------------------------------------------------
# Deterministische Transformationen
# ---------------------------------------------------------------------------

def adjust_brightness(rgb_tuple: Tuple[torch.Tensor, ...], delta: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    return tuple(x + delta for x in rgb_tuple)


def adjust_contrast(rgb_tuple: Tuple[torch.Tensor, ...], factor: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    def _channel(ch: torch.Tensor) -> torch.Tensor:
        mean = ch.mean(dim=(-2, -1), keepdim=True)
        return factor * (ch - mean) + mean
    return tuple(_channel(ch) for ch in rgb_tuple)


def adjust_saturation(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor, factor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return h, torch.clamp(s * factor, 0.0, 1.0), v


def adjust_hue(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor, delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (h + delta) % 1.0, s, v


def _to_grayscale(image: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    gray = (image * weights).sum(dim=-1, keepdim=True)
    return gray.expand_as(image).contiguous()


# ---------------------------------------------------------------------------
# Vmap-Targets (Reine Funktionen für Einzelbilder)
# ---------------------------------------------------------------------------

def _random_flip_pure(image: torch.Tensor, flip_mask: torch.Tensor) -> torch.Tensor:
    # image: [H, W, C], flip_mask: 0D Boolean-Tensor
    flipped = image.flip(dims=[1])  # W-Achse spiegeln (Index 1 bei HWC)
    return torch.where(flip_mask, flipped, image)


def _color_transform_pure(
    image: torch.Tensor,             # [H, W, 3]
    should_apply: torch.Tensor,      # 0D (bool)
    should_apply_gs: torch.Tensor,   # 0D (bool)
    should_apply_cj: torch.Tensor,   # 0D (bool)
    b_delta: torch.Tensor,           # 0D (float)
    c_factor: torch.Tensor,          # 0D (float)
    s_factor: torch.Tensor,          # 0D (float)
    h_delta: torch.Tensor,           # 0D (float)
    order: torch.Tensor,             # [4] (int)
    luma_weights: torch.Tensor,      # [3] (float) - Als Konstante übergeben
    brightness: float, contrast: float, saturation: float, hue: float
) -> torch.Tensor:
    
    rgb = (image[..., 0], image[..., 1], image[..., 2])

    # Statischer Loop über die 4 Transformations-Slots
    for step in range(4):
        op = order[step]

        # 0: Brightness
        r_b, g_b, b_b = adjust_brightness(rgb, b_delta) if brightness > 0 else rgb
        # 1: Contrast
        r_c, g_c, b_c = adjust_contrast(rgb, c_factor) if contrast > 0 else rgb
        
        # HSV-Räume (nur berechnen, wenn verlangt)
        if saturation > 0 or hue > 0:
            h, s, v = rgb_to_hsv(*rgb)
            
            h_s, s_s, v_s = adjust_saturation(h, s, v, s_factor)
            r_s, g_s, b_s = hsv_to_rgb(h_s, s_s, v_s)
            
            h_h, s_h, v_h = adjust_hue(h, s, v, h_delta)
            r_h, g_h, b_h = hsv_to_rgb(h_h, s_h, v_h)
        else:
            r_s, g_s, b_s = rgb
            r_h, g_h, b_h = rgb

        # Vmap-konforme Auswahl via torch.where
        r_next = torch.where(op == 0, r_b, torch.where(op == 1, r_c, torch.where(op == 2, r_s, r_h)))
        g_next = torch.where(op == 0, g_b, torch.where(op == 1, g_c, torch.where(op == 2, g_s, g_h)))
        b_next = torch.where(op == 0, b_b, torch.where(op == 1, b_c, torch.where(op == 2, b_s, b_h)))

        # JAX-konformes direktes Clipping nach jedem Teilschritt
        rgb = (r_next.clamp(0.0, 1.0), g_next.clamp(0.0, 1.0), b_next.clamp(0.0, 1.0))

    out = torch.stack(rgb, dim=-1)

    # Zusammenführung der stochastischen Masken
    out = torch.where(should_apply_cj, out, image)
    
    gray = _to_grayscale(out, luma_weights)
    out = torch.where(should_apply_gs, gray, out)
    
    out = torch.where(should_apply, out, image)
    return out.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Batched Wrapper (Einstiegspunkte)
# ---------------------------------------------------------------------------

@torch.no_grad()
def random_flip(images: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
    gen = torch.Generator(device=images.device)
    if seed is not None:
        gen.manual_seed(seed)

    N = images.shape[0]
    flip_masks = torch.rand(N, generator=gen, device=images.device) <= 0.5

    # Vektorisierung via vmap: Mapping über Dimension 0 für Bild und Maske
    return vmap(_random_flip_pure, in_dims=(0, 0))(images, flip_masks)

# color_jitter.py — proposed:
_CUDA_GENERATORS: dict[torch.device, torch.Generator] = {}

def _get_generator(device: torch.device, seed: int | None) -> torch.Generator:
    key = device
    if key not in _CUDA_GENERATORS:
        _CUDA_GENERATORS[key] = torch.Generator(device=device)
    gen = _CUDA_GENERATORS[key]
    if seed is not None:
        gen.manual_seed(seed)
    return gen


@torch.no_grad()
def color_transform(
    images: torch.Tensor,
    seed: Optional[int] = None,
    brightness: float        = 0.2,
    contrast: float          = 0.1,
    saturation: float        = 0.1,
    hue: float               = 0.03,
    color_jitter_prob: float = 0.8,
    to_grayscale_prob: float = 0.0,
    apply_prob: float        = 1.0,
    shuffle: bool            = True,
) -> torch.Tensor:
    
    five_d = images.dim() == 5
    if five_d:
        images = images[..., 0]

    device = images.device
    dtype = images.dtype
    N = images.shape[0]

    gen = _get_generator(device, seed)

    # 1. Batched RNG-Werte vorab generieren (Pure Inputs für vmap)
    should_apply    = torch.rand(N, generator=gen, device=device, dtype=dtype) <= apply_prob
    should_apply_gs = torch.rand(N, generator=gen, device=device, dtype=dtype) <= to_grayscale_prob
    should_apply_cj = torch.rand(N, generator=gen, device=device, dtype=dtype) <= color_jitter_prob

    b_delta  = (torch.rand(N, generator=gen, device=device, dtype=dtype) * 2.0 - 1.0) * brightness
    c_factor = (torch.rand(N, generator=gen, device=device, dtype=dtype) * 2.0 - 1.0) * contrast + 1.0
    s_factor = (torch.rand(N, generator=gen, device=device, dtype=dtype) * 2.0 - 1.0) * saturation + 1.0
    h_delta  = (torch.rand(N, generator=gen, device=device, dtype=dtype) * 2.0 - 1.0) * hue

    if shuffle:
        order = torch.rand((N, 4), generator=gen, device=device).argsort(dim=1)
    else:
        order = torch.arange(4, device=device).unsqueeze(0).expand(N, 4)

    # Luma-Weights außerhalb erzeugen, um vmap-Fehler zu vermeiden
    luma_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=device, dtype=dtype)

    # 2. in_dims festlegen: 0 = Über Batch-Achse mappen, None = Konstanter Wert für alle
    vmap_in_dims = (
        0,    # image
        0,    # should_apply
        0,    # should_apply_gs
        0,    # should_apply_cj
        0,    # b_delta
        0,    # c_factor
        0,    # s_factor
        0,    # h_delta
        0,    # order
        None, # luma_weights (Konstante)
        None, # brightness (Konstante)
        None, # contrast (Konstante)
        None, # saturation (Konstante)
        None, # hue (Konstante)
    )

    # 3. Vektorisiert ausführen
    vmap_transform = vmap(_color_transform_pure, in_dims=vmap_in_dims)
    
    out = vmap_transform(
        images, should_apply, should_apply_gs, should_apply_cj,
        b_delta, c_factor, s_factor, h_delta, order, luma_weights,
        brightness, contrast, saturation, hue
    )

    if five_d:
        out = out[..., None]

    return out