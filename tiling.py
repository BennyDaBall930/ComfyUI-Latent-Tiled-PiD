"""Tile planning, latent cropping, feather blending and QA math for
Latent-Tiled PiD. Pure numpy/stdlib — no torch, no ComfyUI imports — so it is
unit-testable anywhere: `python tiling.py --self-test`.

Ported verbatim from the validated headless driver
(https://github.com/BennyDaBall930/pid-tiled-decode); keep the two in sync.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field

PID_SCALE = 4  # stage-1 px -> output px
ALIGN = 8      # tile positions/sizes stay multiples of 8 (latent + patch alignment)

GREEN_BIAS_GATE_PTS = 3.0
CHROMA_GATE_PTS = 20.0


@dataclass(frozen=True)
class Tile:
    row: int
    col: int
    x: int      # stage-1 px
    y: int
    w: int
    h: int
    seed: int

    @property
    def name(self) -> str:
        return f"r{self.row}c{self.col}"


@dataclass
class TilePlan:
    image_w: int
    image_h: int
    overlap: int
    xs: list[tuple[int, int]] = field(default_factory=list)
    ys: list[tuple[int, int]] = field(default_factory=list)
    tiles: list[Tile] = field(default_factory=list)

    @property
    def rows(self) -> int:
        return len(self.ys)

    @property
    def cols(self) -> int:
        return len(self.xs)


def _ceil_to(v: float, align: int) -> int:
    return int(math.ceil(v / align)) * align


def plan_axis(size: int, max_tile: int, overlap: int, align: int = ALIGN) -> list[tuple[int, int]]:
    """Cover [0, size) with uniform tiles of size <= max_tile, pairwise overlap
    >= ~overlap. All positions/sizes are multiples of `align`."""
    for name, v in (("size", size), ("max_tile", max_tile), ("overlap", overlap)):
        if v % align:
            raise ValueError(f"{name}={v} must be a multiple of {align}")
    if max_tile <= overlap:
        raise ValueError(f"max_tile ({max_tile}) must exceed overlap ({overlap})")
    if size <= max_tile:
        return [(0, size)]
    n = max(2, math.ceil((size - overlap) / (max_tile - overlap)))
    while True:
        tile = _ceil_to((size + (n - 1) * overlap) / n, align)
        if tile <= max_tile:
            break
        n += 1
    stride = (size - tile) / (n - 1)
    positions = []
    for i in range(n):
        x = int(round(i * stride / align)) * align
        x = min(x, size - tile)
        positions.append((x, tile))
    assert positions[0][0] == 0 and positions[-1][0] + tile == size
    for (a, w), (b, _t) in zip(positions, positions[1:]):
        got = a + w - b
        assert got >= max(align * 2, overlap - align), f"overlap collapsed: {got} at {b}"
    return positions


def make_plan(width: int, height: int, max_tile: int, overlap: int, base_seed: int) -> TilePlan:
    plan = TilePlan(width, height, overlap)
    plan.xs = plan_axis(width, max_tile, overlap)
    plan.ys = plan_axis(height, max_tile, overlap)
    idx = 0
    for r, (y, h) in enumerate(plan.ys):
        for c, (x, w) in enumerate(plan.xs):
            plan.tiles.append(Tile(r, c, x, y, w, h, base_seed + 101 * (idx + 1)))
            idx += 1
    return plan


def crop_latent_tensor(t, x: int, y: int, w: int, h: int):
    """Crop the SPATIAL dims of a latent tensor, layout-agnostic (works on
    numpy arrays and torch tensors, 4-dim [B,C,H,W] or 5-dim [B,C,T,H,W]).

    Core ComfyUI LatentCrop indexes dims 2/3 and silently mangles the 5-dim
    qwen/Wan-family case — always slice the LAST two dims instead."""
    return t[..., y // 8:(y + h) // 8, x // 8:(x + w) // 8]


def axis_weights(pos_sizes: list[tuple[int, int]], scale: int):
    """Per-tile 1D weight vectors (output px). Raised-cosine ramps over the
    exact overlap band with each neighbor; complementary ramps sum to 1."""
    import numpy as np
    weights = []
    for i, (pos, size) in enumerate(pos_sizes):
        w = np.ones(size * scale, dtype=np.float32)
        if i > 0:
            prev_pos, prev_size = pos_sizes[i - 1]
            band = (prev_pos + prev_size - pos) * scale
            if band > 0:
                u = (np.arange(band, dtype=np.float32) + 0.5) / band
                w[:band] = 0.5 - 0.5 * np.cos(np.pi * u)
        if i < len(pos_sizes) - 1:
            next_pos, _next_size = pos_sizes[i + 1]
            band = (pos + size - next_pos) * scale
            if band > 0:
                u = (np.arange(band, dtype=np.float32) + 0.5) / band
                w[-band:] = np.minimum(w[-band:], (0.5 + 0.5 * np.cos(np.pi * u)))
        weights.append(w)
    return weights


def blend_tiles(plan: TilePlan, tile_images: dict, scale: int = PID_SCALE):
    """Feather-blend {tile.name: float32 [H,W,3] in [0,1]} into the full image."""
    import numpy as np
    W, H = plan.image_w * scale, plan.image_h * scale
    acc = np.zeros((H, W, 3), dtype=np.float32)
    den = np.zeros((H, W, 1), dtype=np.float32)
    wx = axis_weights(plan.xs, scale)
    wy = axis_weights(plan.ys, scale)
    for tile in plan.tiles:
        img = tile_images[tile.name]
        th, tw = tile.h * scale, tile.w * scale
        if img.shape[:2] != (th, tw):
            raise ValueError(f"tile {tile.name} is {img.shape[1]}x{img.shape[0]}, expected {tw}x{th}")
        w2d = (wy[tile.row][:, None] * wx[tile.col][None, :]).astype(np.float32)[..., None]
        ys, xs = tile.y * scale, tile.x * scale
        acc[ys:ys + th, xs:xs + tw] += img * w2d
        den[ys:ys + th, xs:xs + tw] += w2d
    if float(den.min()) <= 0:
        raise RuntimeError("blend hole: some output pixels are covered by no tile")
    return acc / den


def shadow_green_bias(img, luma_thresh: float = 0.24) -> float:
    """Mean (G - (R+B)/2) over shadow pixels, 0-255 scale. img float [0,1]."""
    luma = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    mask = luma < luma_thresh
    if not bool(mask.any()):
        return float("nan")
    bias = img[..., 1][mask] - 0.5 * (img[..., 0][mask] + img[..., 2][mask])
    return float(bias.mean() * 255.0)


def midtone_chroma_delta(img, ref) -> float:
    """Mean opponent-chroma distance vs ref over ref midtones (0-255 scale).
    Catches the past-envelope PiD collapse (midtone magenta / desaturation)
    that a shadow-green mean misses. img must match ref's resolution."""
    import numpy as np
    luma = 0.2126 * ref[..., 0] + 0.7152 * ref[..., 1] + 0.0722 * ref[..., 2]
    mask = (luma > 0.25) & (luma < 0.8)
    if not bool(mask.any()):
        return float("nan")

    def opp(x):
        return x[..., 0] - x[..., 1], 0.5 * (x[..., 0] + x[..., 1]) - x[..., 2]

    rg_a, yb_a = opp(img)
    rg_b, yb_b = opp(ref)
    d = np.sqrt((rg_a - rg_b) ** 2 + (yb_a - yb_b) ** 2)
    return float(d[mask].mean() * 255.0)


def self_test() -> int:
    import numpy as np
    cases = [(1344, 768, 512, 64), (1920, 1088, 1024, 64), (2048, 1152, 1024, 64),
             (4096, 2304, 1024, 128), (768, 768, 1024, 64)]
    for w, h, mt, ov in cases:
        plan = make_plan(w, h, mt, ov, 1)
        for (pos, size) in plan.xs + plan.ys:
            assert pos % ALIGN == 0 and size % ALIGN == 0
        assert plan.xs[-1][0] + plan.xs[-1][1] == w
        assert plan.ys[-1][0] + plan.ys[-1][1] == h
        print(f"[plan] {w}x{h} max={mt} ov={ov} -> {plan.cols}x{plan.rows}")
    plan = make_plan(1344, 768, 512, 64, 1)
    W, H = plan.image_w * PID_SCALE, plan.image_h * PID_SCALE
    gx = np.linspace(0, 1, W, dtype=np.float32)[None, :, None]
    gy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    world = np.concatenate([gx * np.ones((H, 1, 1), np.float32),
                            gy * np.ones((1, W, 1), np.float32),
                            0.25 * np.ones((H, W, 1), np.float32)], axis=2)
    tiles = {t.name: world[t.y * PID_SCALE:(t.y + t.h) * PID_SCALE,
                           t.x * PID_SCALE:(t.x + t.w) * PID_SCALE].copy()
             for t in plan.tiles}
    out = blend_tiles(plan, tiles)
    assert float(np.abs(out - world).max()) < 1e-4
    lat5 = np.arange(2 * 16 * 1 * 96 * 168).reshape(2, 16, 1, 96, 168)
    c = crop_latent_tensor(lat5, x=424, y=352, w=496, h=416)
    assert c.shape == (2, 16, 1, 52, 62), c.shape
    assert (c == lat5[:, :, :, 44:96, 53:115]).all()
    gray = np.full((64, 64, 3), 0.1, np.float32)
    assert abs(shadow_green_bias(gray)) < 1e-6
    tinted = gray.copy()
    tinted[..., 1] += 0.02
    assert shadow_green_bias(tinted) > 3.0
    assert midtone_chroma_delta(np.full((64, 64, 3), 0.5, np.float32),
                                np.full((64, 64, 3), 0.5, np.float32)) < 1e-6
    print("SELF-TEST PASS")
    return 0


if __name__ == "__main__":
    sys.exit(self_test() if "--self-test" in sys.argv else 0)
