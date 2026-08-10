"""Latent-Tiled PiD — ComfyUI nodes.

Tile the latent, not the pixels: decodes a generation latent through NVIDIA's
PiD (Pixel Diffusion Decoder) as overlapping latent tiles, each a normal-sized
PiD job inside the model's trained 1024->4096 envelope, then feather-blends the
pixel tiles. Single-shot PiD past ~4K output suffers midtone color collapse;
tiling the latent avoids it with no VAE re-encode round-trip.

Companion headless driver + measurements:
https://github.com/BennyDaBall930/pid-tiled-decode
"""
from __future__ import annotations

import numpy as np
import torch

import comfy.latent_formats
import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
import node_helpers

from .tiling import (
    ALIGN,
    ALIGN_FLUX2,
    PID_SCALE,
    PRESETS,
    PRESETS_FLUX2,
    blend_tiles,
    crop_latent_tensor,
    make_plan,
    midtone_chroma_delta,
    shadow_green_bias,
)

# The 4-step DMD2-distilled schedule the released PiD checkpoints were trained
# for (do not respace) + the lcm sampler, matching the reference workflow.
PID_SIGMAS = [0.999, 0.866, 0.634, 0.342, 0.0]


def _zero_out(conditioning):
    """Core ConditioningZeroOut, inlined (avoids importing the server's nodes
    module): zero the cond tensor and pooled_output."""
    out = []
    for t in conditioning:
        d = t[1].copy()
        pooled = d.get("pooled_output", None)
        if pooled is not None:
            d["pooled_output"] = torch.zeros_like(pooled)
        out.append([torch.zeros_like(t[0]), d])
    return out


def _latent_format(name: str, channels: int):
    if name in ("flux", "flux2"):
        flux2 = getattr(comfy.latent_formats, "Flux2", None)
        if name == "flux2" or (channels == 128 and flux2 is not None):
            if flux2 is None:
                raise RuntimeError(
                    "latent_format 'flux2' needs a ComfyUI version with FLUX.2 "
                    "support — update ComfyUI and retry.")
            return flux2()
        return comfy.latent_formats.Flux()
    if name == "sd3":
        return comfy.latent_formats.SD3()
    if name == "sdxl":
        return comfy.latent_formats.SDXL()
    if name == "qwenimage":
        return comfy.latent_formats.Wan21()
    raise ValueError(f"Unknown latent_format: {name}")


def _pixel_space_vae():
    try:
        vae = comfy.sd.VAE(sd={"pixel_space_vae": torch.tensor(1.0)})
        if hasattr(vae, "throw_exception_if_invalid"):
            vae.throw_exception_if_invalid()
        return vae
    except Exception as exc:
        raise RuntimeError(
            "Could not construct the pixel-space VAE — Latent-Tiled PiD needs a ComfyUI "
            "version with PiD support (>= 0.28). Update ComfyUI and retry."
        ) from exc


def _fix_channels(model, empty):
    try:
        return comfy.sample.fix_empty_latent_channels(model, empty)
    except Exception:
        return empty  # pixel-space latents are already 3ch; safe to proceed


class LatentTiledPiDDecode:
    """LATENT in -> high-res IMAGE out, via latent-native tiled PiD decoding."""

    SEARCH_ALIASES = ["pid tiled", "tiled pid", "latent tiled upscale"]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL", {"tooltip": "The PiD checkpoint (UNETLoader)."}),
            "positive": ("CONDITIONING", {"tooltip": "PiD text conditioning (Gemma CLIPTextEncode; empty prompt is fine)."}),
            "latent": ("LATENT", {"tooltip": "The generation latent straight from your KSampler (flux/flux2-klein/sd3/sdxl/qwen-family)."}),
            "latent_format": (["flux", "flux2", "sd3", "sdxl", "qwenimage"],
                              {"default": "qwenimage",
                               "tooltip": "Family of the STAGE-1 latent. 'flux2' = FLUX.2 dev + Klein 4B/9B "
                                          "(128-ch, 16x); Flux2 is also auto-detected under 'flux' from the "
                                          "channel count, matching core PiDConditioning."}),
            "max_tile": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 8,
                                 "tooltip": "Max tile size per axis in stage-1 px. 1024 = PiD's trained envelope."}),
            "overlap": ("INT", {"default": 64, "min": 16, "max": 512, "step": 8,
                                "tooltip": "Tile overlap in stage-1 px (x4 in output; feather band)."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                             "control_after_generate": True}),
            "degrade_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "PiD noisy-latent conditioning: raise for partially-denoised latents."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "latent/pid"
    DESCRIPTION = ("Decodes the latent as overlapping tiles through PiD — every tile stays inside "
                   "the trained 1024->4096 envelope, avoiding the color collapse of single-shot "
                   "decoding at large sizes — then feather-blends (seam-free, weights sum to 1).")

    def decode(self, model, positive, latent, latent_format, max_tile, overlap, seed, degrade_sigma):
        samples = latent["samples"]
        fmt = _latent_format(latent_format, samples.shape[1])
        # Family geometry: one latent cell = `down` stage-1 px (8 for the 8x
        # families, 16 for FLUX.2/Klein). Tile positions/sizes must land on
        # whole latent cells, so the plan aligns to `down`.
        down = int(getattr(fmt, "spacial_downscale_ratio", 8) or 8)
        align = max(ALIGN, down)
        mt = max(align * 2, max_tile - max_tile % align)
        ov = max(align, overlap - overlap % align)
        if (mt, ov) != (max_tile, overlap):
            print(f"[Latent-Tiled PiD] snapped max_tile/overlap {max_tile}/{overlap} "
                  f"-> {mt}/{ov} (multiples of {align} for this latent family)")
        stage1_w, stage1_h = samples.shape[-1] * down, samples.shape[-2] * down
        plan = make_plan(stage1_w, stage1_h, mt, ov, seed, align=align)
        print(f"[Latent-Tiled PiD] {stage1_w}x{stage1_h} latent -> "
              f"{plan.cols}x{plan.rows} tiles ({len(plan.tiles)}) of "
              f"{plan.xs[0][1] * PID_SCALE}x{plan.ys[0][1] * PID_SCALE} out-px -> "
              f"{stage1_w * PID_SCALE}x{stage1_h * PID_SCALE} output")

        negative = _zero_out(positive)
        sampler = comfy.samplers.sampler_object("lcm")
        sigmas = torch.FloatTensor(PID_SIGMAS)
        vae = _pixel_space_vae()
        sigma_t = torch.tensor([float(degrade_sigma)], dtype=torch.float32)

        pbar = comfy.utils.ProgressBar(len(plan.tiles))
        tile_images: dict[str, np.ndarray] = {}
        batch = samples.shape[0]
        for tile in plan.tiles:
            comfy.model_management.throw_exception_if_processing_interrupted()
            crop = crop_latent_tensor(samples, tile.x, tile.y, tile.w, tile.h, down=down)
            lq = fmt.process_in(crop)
            if lq.ndim == 5:
                lq = lq[:, :, 0]
            cond = node_helpers.conditioning_set_values(
                positive, {"lq_latent": lq, "degrade_sigma": sigma_t})

            empty = torch.zeros((batch, 3, tile.h * PID_SCALE, tile.w * PID_SCALE),
                                device=comfy.model_management.intermediate_device())
            empty = _fix_channels(model, empty)
            noise = comfy.sample.prepare_noise(empty, tile.seed, None)
            out = comfy.sample.sample_custom(
                model, noise, 1.0, sampler, sigmas, cond, negative, empty,
                noise_mask=None, callback=None, disable_pbar=True, seed=tile.seed)

            img = vae.decode(out)  # [B, H, W, 3] in [0, 1]
            tile_images[tile.name] = img[0].to(torch.float32).cpu().numpy() if batch == 1 else \
                img.to(torch.float32).cpu().numpy()
            pbar.update(1)

        if batch == 1:
            blended = blend_tiles(plan, tile_images)
            image = torch.from_numpy(blended)[None, ...]
        else:
            per_b = []
            for b in range(batch):
                blended = blend_tiles(plan, {k: v[b] for k, v in tile_images.items()})
                per_b.append(torch.from_numpy(blended))
            image = torch.stack(per_b, dim=0)
        return (image,)


class LatentTiledPiDSize:
    """One dropdown, zero math: every entry shows aspect, render size, output
    size, megapixels and tile count. Emits the empty latent at that size —
    wire it where EmptySD3LatentImage would go."""

    SEARCH_ALIASES = ["pid size", "poster size", "tile size picker"]

    _MAP = {label: (w, h, tiles) for label, w, h, tiles in PRESETS}
    _DEFAULT = next(label for label, w, h, tiles in PRESETS if (w, h) == (1920, 1088))

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "size": ([label for label, _w, _h, _t in PRESETS],
                     {"default": cls._DEFAULT,
                      "tooltip": "aspect | stage-1 render size -> final output size | megapixels | "
                                 "tile count. Pick a line; there is nothing else to configure."}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "latent/pid"
    DESCRIPTION = ("Every dropdown entry is a complete, planner-validated configuration: the "
                   "aspect ratio, the stage-1 render size this latent will have, the output "
                   "resolution the tiled decode will produce, and how many tiles it takes. "
                   "All entries keep every tile inside PiD's trained envelope.")

    def generate(self, size, batch_size):
        if size not in self._MAP:
            raise ValueError(f"Unknown size preset: {size!r} — re-select from the dropdown "
                             "(the preset list may have changed between versions).")
        w, h, tiles = self._MAP[size]
        print(f"[Latent-Tiled PiD Size] {size}")
        # Mirrors core EmptySD3LatentImage.execute (comfy_extras/nodes_sd3.py) —
        # it is a v3 io.ComfyNode, not callable classic-style from here.
        dtype_fn = getattr(comfy.model_management, "intermediate_dtype", None)
        latent = torch.zeros(
            [batch_size, 16, h // 8, w // 8],
            device=comfy.model_management.intermediate_device(),
            dtype=dtype_fn() if dtype_fn is not None else torch.float32)
        return ({"samples": latent, "downscale_ratio_spacial": 8},)


class LatentTiledPiDSizeFlux2:
    """Size picker for the FLUX.2 family (dev + Klein 4B/9B): every entry is
    16-grid-aligned for the 128-ch, 16x-downscale flux2 latent space. Emits the
    empty flux2 latent plus width/height INTs to wire into Flux2Scheduler."""

    SEARCH_ALIASES = ["pid size flux2", "klein size", "flux2 tile size picker"]

    _MAP = {label: (w, h, tiles) for label, w, h, tiles in PRESETS_FLUX2}
    _DEFAULT = next(label for label, w, h, tiles in PRESETS_FLUX2 if (w, h) == (1920, 1088))

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "size": ([label for label, _w, _h, _t in PRESETS_FLUX2],
                     {"default": cls._DEFAULT,
                      "tooltip": "aspect | stage-1 render size -> final output size | megapixels | "
                                 "tile count. Pick a line; there is nothing else to configure."}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
        }}

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "width", "height")
    FUNCTION = "generate"
    CATEGORY = "latent/pid"
    DESCRIPTION = ("FLUX.2 / Klein flavor of the Size Picker: emits the native 128-channel "
                   "16x-downscale empty latent at a planner-validated size, plus the matching "
                   "width/height (wire those into Flux2Scheduler so its resolution-dependent "
                   "shift tracks the preset). All entries keep every tile inside PiD's "
                   "trained envelope.")

    def generate(self, size, batch_size):
        if size not in self._MAP:
            raise ValueError(f"Unknown size preset: {size!r} — re-select from the dropdown "
                             "(the preset list may have changed between versions).")
        w, h, tiles = self._MAP[size]
        print(f"[Latent-Tiled PiD Size Flux2] {size}")
        # Mirrors core EmptyFlux2LatentImage (comfy_extras/nodes_flux.py).
        dtype_fn = getattr(comfy.model_management, "intermediate_dtype", None)
        latent = torch.zeros(
            [batch_size, 128, h // ALIGN_FLUX2, w // ALIGN_FLUX2],
            device=comfy.model_management.intermediate_device(),
            dtype=dtype_fn() if dtype_fn is not None else torch.float32)
        return ({"samples": latent, "downscale_ratio_spacial": ALIGN_FLUX2}, w, h)


class LatentTiledPiDQA:
    """Compare a PiD decode against its VAE-decode twin: midtone chroma delta
    (catches the past-envelope color collapse; healthy ~8-12 on PiD v1.5,
    10-15 on v1, broken ~30, flag > 20) and shadow green-bias delta (flag +/-3)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE", {"tooltip": "The PiD decode (any resolution)."}),
            "reference": ("IMAGE", {"tooltip": "The stage-1 VAE decode of the same latent."}),
        }}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "measure"
    CATEGORY = "latent/pid"
    OUTPUT_NODE = True

    def measure(self, image, reference):
        img = image[0].to(torch.float32).cpu().numpy()
        ref = reference[0].to(torch.float32).cpu().numpy()
        if img.shape[:2] != ref.shape[:2]:
            t = torch.from_numpy(img).permute(2, 0, 1)[None]
            t = torch.nn.functional.interpolate(t, size=ref.shape[:2], mode="area")
            img = t[0].permute(1, 2, 0).numpy()
        chroma = midtone_chroma_delta(img, ref)
        green = shadow_green_bias(img) - shadow_green_bias(ref)
        chroma_ok = chroma <= 20.0
        green_ok = abs(green) <= 3.0
        text = (f"midtone chroma delta: {chroma:.2f} pts "
                f"({'PASS' if chroma_ok else 'CHECK — color collapse territory'}; flag > 20)\n"
                f"shadow green-bias delta: {green:+.2f} pts "
                f"({'PASS' if green_ok else 'CHECK'}; flag +/-3)")
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "LatentTiledPiDDecode": LatentTiledPiDDecode,
    "LatentTiledPiDSize": LatentTiledPiDSize,
    "LatentTiledPiDSizeFlux2": LatentTiledPiDSizeFlux2,
    "LatentTiledPiDQA": LatentTiledPiDQA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentTiledPiDDecode": "Latent-Tiled PiD Decode",
    "LatentTiledPiDSize": "Latent-Tiled PiD Size Picker",
    "LatentTiledPiDSizeFlux2": "Latent-Tiled PiD Size Picker (FLUX.2 / Klein)",
    "LatentTiledPiDQA": "Latent-Tiled PiD QA (vs VAE twin)",
}
