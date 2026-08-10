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
ALIGN_FLUX2 = 16  # FLUX.2 / Klein latents are 16x-downscaled: one latent cell = 16 stage-1 px

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


def make_plan(width: int, height: int, max_tile: int, overlap: int, base_seed: int,
              align: int = ALIGN) -> TilePlan:
    plan = TilePlan(width, height, overlap)
    plan.xs = plan_axis(width, max_tile, overlap, align=align)
    plan.ys = plan_axis(height, max_tile, overlap, align=align)
    idx = 0
    for r, (y, h) in enumerate(plan.ys):
        for c, (x, w) in enumerate(plan.xs):
            plan.tiles.append(Tile(r, c, x, y, w, h, base_seed + 101 * (idx + 1)))
            idx += 1
    return plan


def solve_size(tiles: int, aspect_w: int, aspect_h: int,
               max_tile: int = 1024, overlap: int = 64, align: int = ALIGN):
    """Largest stage-1 (width, height, cols, rows) whose plan yields exactly
    `tiles` tiles at the given aspect, every tile inside the envelope.
    Returns None when the combo is impossible (e.g. 6 tiles at 1:1)."""
    def axis_max(n: int) -> int:
        return n * max_tile - (n - 1) * overlap

    best = None
    for cols in range(1, tiles + 1):
        if tiles % cols:
            continue
        rows = tiles // cols
        w = min(axis_max(cols), axis_max(rows) * aspect_w / aspect_h)
        w = (int(w) // align) * align
        h = (int(w * aspect_h / aspect_w) // align) * align
        if w <= 0 or h <= 0 or h > axis_max(rows):
            continue
        if cols > 1 and w <= axis_max(cols - 1):
            continue
        if rows > 1 and h <= axis_max(rows - 1):
            continue
        if best is None or w * h > best[0] * best[1]:
            best = (w, h, cols, rows)
    return best


# Render-proven ladder sizes (16:9 family, validated 2026-08-04/06 incl. the
# 137MP and 213MP rungs) take precedence over solver maximums so the dropdown
# matches the tested sizes exactly. Sub-1024 single-tile rungs were tested and
# REJECTED: stage-1 models go mushy below their trained res and PiD reinterprets
# instead of decoding (chroma gate flags it) — the 1-tile ~1024-class entry is
# the deliberate floor.
_PROVEN = {
    (16, 9): {2: (1344, 768), 4: (1920, 1088), 6: (2560, 1440), 8: (3456, 1944),
              12: (3904, 2192), 15: (4864, 2736)},
    (9, 16): {2: (768, 1344), 4: (1088, 1920), 6: (1440, 2560), 8: (1944, 3456),
              12: (2192, 3904), 15: (2736, 4864)},
}

_ASPECT_ORDER = [("16:9", 16, 9), ("9:16", 9, 16), ("4:3", 4, 3), ("3:4", 3, 4),
                 ("3:2", 3, 2), ("2:3", 2, 3), ("4:5", 4, 5), ("5:4", 5, 4),
                 ("1:1", 1, 1)]

_TILE_TIERS = (2, 4, 6, 8, 9, 12, 15, 16)

# v1.1.1 culled the three ~210MP 15-tile rungs (chroma just past the gate on
# the v1 checkpoint). PiD v1.5's color-fidelity fix redeemed all three on
# retest (chroma 9.8-11.8 vs 20.1-21.1) — restored in v1.2.0 with the v1.5
# checkpoint as the shipped default. Empty dict kept as the cull mechanism.
_CULLED: dict = {}

# flux2-family culls (2026-08-10 Klein 4B sweep): the 4:3 6-tile rung failed
# the shadow green-bias gate twice (-3.66, then -5.15 on a seed-shift retest) —
# systematic, so it doesn't ship. Chroma was healthy both times; candidate for
# restoration if a future PiD flux2 checkpoint fixes the shadow drift. The 4:5
# 9-tile rung flagged once (-4.03) but passed its retest (-0.66) — kept.
_CULLED_FLUX2: dict = {(4, 3): {6}}


def _single_tile_size(aw: int, ah: int, max_tile: int = 1024, align: int = ALIGN):
    if aw >= ah:
        w = max_tile
        h = (int(w * ah / aw) // align) * align
    else:
        h = max_tile
        w = (int(h * aw / ah) // align) * align
    return w, h


def build_presets(align: int = ALIGN, culled: dict | None = None):
    """[(label, w, h, tiles)] — every entry planner-validated by the self-test.
    Labels show aspect, render size, output size, megapixels and tile count so
    the dropdown itself is the whole setup guide. `align` is the family's
    latent-cell size in stage-1 px (8 for the 8x families, 16 for FLUX.2);
    proven ladder sizes are reused only when they land on the family's grid,
    otherwise the solver supplies the nearest in-envelope size. `culled` maps
    (aspect_w, aspect_h) -> set of tile tiers that failed that family's QA
    gates and must not ship."""
    if culled is None:
        culled = _CULLED
    presets = []

    def label(name, w, h, tiles):
        ow, oh = w * PID_SCALE, h * PID_SCALE
        mp = ow * oh / 1e6
        t = "1 tile" if tiles == 1 else f"{tiles} tiles"
        return f"{name} | render {w}x{h} -> output {ow}x{oh} | {mp:.0f} MP | {t}"

    for name, aw, ah in _ASPECT_ORDER:
        w, h = _single_tile_size(aw, ah, align=align)
        presets.append((label(name, w, h, 1), w, h, 1))
        prev_area = w * h
        for tiles in _TILE_TIERS:
            if tiles in culled.get((aw, ah), set()):
                continue
            proven = _PROVEN.get((aw, ah), {}).get(tiles)
            if proven is not None and proven[0] % align == 0 and proven[1] % align == 0:
                w, h = proven
            else:
                solved = solve_size(tiles, aw, ah, align=align)
                if solved is None:
                    continue  # impossible combos never reach the dropdown
                w, h, _c, _r = solved
            if w * h < prev_area * 1.05:
                continue  # skip rungs that add tiles without adding real size
            prev_area = w * h
            presets.append((label(name, w, h, tiles), w, h, tiles))
    return presets


PRESETS = build_presets()
PRESETS_FLUX2 = build_presets(align=ALIGN_FLUX2, culled=_CULLED_FLUX2)


def crop_latent_tensor(t, x: int, y: int, w: int, h: int, down: int = 8):
    """Crop the SPATIAL dims of a latent tensor, layout-agnostic (works on
    numpy arrays and torch tensors, 4-dim [B,C,H,W] or 5-dim [B,C,T,H,W]).
    Coordinates are stage-1 px; `down` is the family's spatial downscale
    (8 for flux1/sd3/sdxl/qwen, 16 for FLUX.2/Klein).

    Core ComfyUI LatentCrop indexes dims 2/3 and silently mangles the 5-dim
    qwen/Wan-family case — always slice the LAST two dims instead."""
    return t[..., y // down:(y + h) // down, x // down:(x + w) // down]


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
    # presets: every dropdown entry must round-trip through the planner to
    # EXACTLY its stated tile count, all tiles in-envelope, dims aligned
    for family, plist, align in (("8x", PRESETS, ALIGN),
                                 ("flux2", PRESETS_FLUX2, ALIGN_FLUX2)):
        seen_labels = set()
        for lbl, w, h, tiles in plist:
            assert lbl not in seen_labels, f"duplicate preset label: {lbl}"
            seen_labels.add(lbl)
            assert w % align == 0 and h % align == 0, f"{family}: {lbl}"
            if tiles == 1:
                assert w <= 1024 and h <= 1024, lbl
            else:
                plan = make_plan(w, h, 1024, 64, 1, align=align)
                assert len(plan.tiles) == tiles, \
                    f"{family} {lbl}: promised {tiles} tiles, planner gives {len(plan.tiles)}"
                assert plan.xs[0][1] <= 1024 and plan.ys[0][1] <= 1024, lbl
                for (pos, size) in plan.xs + plan.ys:
                    assert pos % align == 0 and size % align == 0, f"{family}: {lbl}"
            print(f"[preset {family}] {lbl}")
    assert solve_size(6, 1, 1) is None and solve_size(8, 1, 1) is None
    # flux2 alignment: 16x plans land every boundary on a latent cell
    plan16 = make_plan(1920, 1088, 1024, 64, 1, align=ALIGN_FLUX2)
    for (pos, size) in plan16.xs + plan16.ys:
        assert pos % 16 == 0 and size % 16 == 0
    # flux2 crop: px coords -> 16x latent cells, exact slices
    lat2 = np.arange(1 * 128 * 68 * 120).reshape(1, 128, 68, 120)
    c16 = crop_latent_tensor(lat2, x=448, y=352, w=512, h=416, down=16)
    assert c16.shape == (1, 128, 26, 32), c16.shape
    assert (c16 == lat2[:, :, 22:48, 28:60]).all()
    # the flagship proven rungs survive on the 16-grid with identical sizes
    flux2_sizes = {(w, h) for _l, w, h, _t in PRESETS_FLUX2}
    for wh in ((1344, 768), (1920, 1088), (2560, 1440), (3904, 2192), (4864, 2736)):
        assert wh in flux2_sizes, f"flux2 presets lost proven rung {wh}"
    print(f"SELF-TEST PASS ({len(PRESETS)} presets 8x, {len(PRESETS_FLUX2)} presets flux2)")
    return 0


if __name__ == "__main__":
    sys.exit(self_test() if "--self-test" in sys.argv else 0)
