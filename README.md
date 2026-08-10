# ComfyUI-Latent-Tiled-PiD

**Tile the latent, not the pixels.**

[![self-test](https://github.com/BennyDaBall930/ComfyUI-Latent-Tiled-PiD/actions/workflows/selftest.yml/badge.svg)](https://github.com/BennyDaBall930/ComfyUI-Latent-Tiled-PiD/actions/workflows/selftest.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Latent-native tiled decoding for NVIDIA [PiD](https://research.nvidia.com/labs/sil/projects/pid/)
(Pixel Diffusion Decoder, [arXiv:2605.23902](https://arxiv.org/abs/2605.23902)) as native ComfyUI
nodes. One node between KSampler and SaveImage: slices the **generation latent** into overlapping
tiles, decodes every tile as a normal-sized in-envelope PiD job, feather-blends the pixels. Seam-free
33MP from a 2MP latent in about a minute — verified up to **244MP** (15616x15616), with **56
render-tested sizes across 9 aspect ratios** and cost scaling ~linearly at roughly **1MP of
finished output per second on a single RTX 5090**.

**v1.3.0 adds the FLUX.2 family — Klein 4B, Klein 9B and FLUX.2-dev** (128-channel, 16x-downscale
latents) with its own dedicated Size Picker and its own **55-entry preset ladder, render-tested
the same way**: all 56 candidate rungs swept end-to-end on Klein 4B distilled up to 244MP, 55
shipped, one culled on a repeat QA-gate failure
([docs/preset_sweep_report_flux2.md](docs/preset_sweep_report_flux2.md)).

![33MP tiled decode](images/hero_tiled_33mp.jpg)

## Why this exists

Single-shot PiD decoding is only trained for the 1024→4096 envelope. Push a 2MP latent through it
and midtone color collapses — whites turn pink, chroma noise everywhere. The model isn't broken;
you're just asking it to perform outside its contract, and it answers in magenta:

![single-shot collapse vs tiled — this 100% crop sits directly on a tile seam](images/ab_collapse_roses.jpg)

That crop sits **directly on the vertical tile seam** — find it. And the same decode that fixes the
color is a genuine detail upgrade over the plain VAE decode:

![real synthesized detail vs the VAE decode](images/ab_detail_dahlia.jpg)

Existing tiled-PiD flows tile the **decoded image** and VAE-encode each tile back to latent space
before running PiD — a lossy round-trip through tiles that were never real latents. Tiling the
generation latent itself skips all of that: every tile conditions on a bit-exact crop of the latent
the sampler actually produced, PiD's sigma-gated LQ injection keeps tiles globally coherent, and the
complementary raised-cosine feather (weights sum to exactly 1) leaves no seams (worst measured seam
gradient ratio across all test runs: 0.77–1.04, where 1.0 is statistically invisible).

The idea came from a simple observation while measuring the collapse: PiD's RoPE is relative, so a
cropped latent decoded as its own image is indistinguishable from a native render of that size. I
wanted the decode path the paper actually describes — latent in, pixels out — at sizes the single
shot can't reach. This pack is that, as two nodes.

## Nodes

**The one rule to internalize: output resolution = 4x your latent, always.** Feed a bigger latent
and the node adds tiles automatically — there is nothing to configure. 1MP latent → 1 tile → 16MP.
2MP → 4 tiles → 33MP. 6.7MP → 8 tiles → 107MP. 15.2MP → 16 tiles → 244MP. Same node, same settings.

### Latent-Tiled PiD Size Picker

`(nothing) -> LATENT` — **start here.** One dropdown, zero math. Wire it where
`EmptySD3LatentImage` would go and pick a line:

```
16:9 | render 1920x1088 -> output 7680x4352 | 33 MP | 4 tiles
```

Every entry is a complete configuration: aspect, the stage-1 render size this latent will have, the
exact output resolution the tiled decode will produce, and how many tiles it takes. **56 presets
across 9 aspect ratios, every single one render-tested end-to-end** — full QA sweep on Krea 2 Turbo,
per-preset verdicts in [docs/preset_sweep_report.md](docs/preset_sweep_report.md). Range: single-tile
4K-class up to **244 MP** (1:1, 15616x15616). Three ~210MP 15-tile rungs briefly shipped culled
(their chroma tested just past the gate on the v1 checkpoint) and were restored in v1.2.0 when
**PiD v1.5's color-fidelity fix cut their chroma roughly in half on retest** (9.8–11.8 vs
20.1–21.1). There is deliberately no sub-1024 single-tile entry: below their trained res, stage-1
models go mushy and PiD reinterprets instead of decoding — the ~1024 single-tile entry is the floor
on purpose. Nothing in this dropdown is theoretical.

> **Model support:** two render-tested lanes. **Krea 2 Turbo** (qwen-family, 8x latents) — the
> original full sweep — and, since v1.3.0, **FLUX.2 Klein 4B distilled** (flux2 family, 16x
> latents) with its own full 56-preset sweep
> ([docs/preset_sweep_report_flux2.md](docs/preset_sweep_report_flux2.md)). **Klein 9B (fp8):
> spot-validated clean through the 59MP rungs** (chroma 10.2–13.9) but degrades progressively from
> ~111MP up — shadow drift first, then chroma past the gate at 137MP+
> ([docs/preset_sweep_report_flux2_9b.md](docs/preset_sweep_report_flux2_9b.md)); **use 4B for the
> monster rungs**, which sweeps clean to 244MP. Other qwen-family stage-1 models should behave
> like Krea 2; flux1/sd3/sdxl paths exist in the Decode node but are untested. If you run
> something else, run the QA node and trust its verdicts over hope.

### Latent-Tiled PiD Size Picker (FLUX.2 / Klein) — v1.3.0

The FLUX.2 twin of the Size Picker: same one-dropdown contract, but every entry sits on the
**16-px grid** the flux2 latent space requires (a handful of rung sizes differ slightly from the
8x ladder for that reason — e.g. the 8-tile 16:9 rung is 3520x1968 here vs 3456x1944 there).
Three outputs instead of one: the empty **128-channel flux2 latent**, plus **width/height INTs —
wire those two into `Flux2Scheduler`** so its resolution-dependent sigma shift always matches the
preset. Stage-1 recipe for Klein distilled (4B and 9B): `Flux2Scheduler` steps=4 → euler via
`KSamplerSelect` → `CFGGuider` cfg=1.0 → `SamplerCustomAdvanced`, exactly as in the shipped
example workflows.

### Latent-Tiled PiD Decode

`MODEL + CONDITIONING + LATENT -> IMAGE`

| input | notes |
|---|---|
| `model` | PiD checkpoint via `UNETLoader` — use **v1.5** for your family: `pid_1.5_qwenimage_…` (qwen), `pid_1.5_flux2_1024_to_4096_4step_bf16.safetensors` (FLUX.2 dev + Klein 4B/9B), `pid_1.5_flux1_…` (flux1); v1 works but is deprecated upstream and measurably worse on color |
| `positive` | Gemma-2 `CLIPTextEncode` (CLIPLoader type `pixeldit`); empty prompt works — same encoder for every family |
| `latent` | generation latent straight off the sampler — flux / **flux2 (Klein)** / sd3 / sdxl / qwen-family |
| `latent_format` | `qwenimage` default; **`flux2` for FLUX.2 dev / Klein** (also auto-detected from the 128-channel count under `flux`, matching core `PiDConditioning`) |
| `max_tile` | max tile edge in stage-1 px; **1024 = the trained envelope, leave it there** |
| `overlap` | feather band in stage-1 px (x4 in output); 64 default |
| `seed` | tile i noises with `seed + 101*(i+1)` |
| `degrade_sigma` | PiD noisy-latent conditioning for early-exit latents; 0.0 for clean latents |

Runs the checkpoint's distilled 4-step schedule (`0.999, 0.866, 0.634, 0.342, 0`) with `lcm` at
cfg 1.0 internally — the reference PiD configuration; do not expect other schedules to work on a
DMD2-distilled student. Per-tile progress bar, interruptible, no temp files.

### There is no "tile mode" switch — the latent size is the mode

Tile count is derived, not selected: output is always **4x the latent**, and the planner grows the
grid to cover whatever you feed it while keeping every tile inside the envelope. **The Size Picker
node's dropdown IS this table** — use it and skip the math. If you'd rather use a stock
`EmptySD3LatentImage`, set it to one of these (all render-proven on Krea 2 Turbo) and you get the
tile count in the left column:

| tiles | landscape latent | portrait latent | output (landscape) | megapixels |
|---|---|---|---|---|
| 1 | up to 1024x1024 | up to 1024x1024 | up to 4096x4096 | ≤ 16.8 |
| 4 (2x2) | **1920x1088** | 1088x1920 | 7680x4352 | 33.4 |
| 6 (3x2) | **2560x1440** | 1440x2560 | 10240x5760 | 59.0 |
| 8 (4x2) | **3456x1944** | 1944x3456 | 13824x7776 | 107.5 |
| 12 (4x3) | **3904x2192** | 2192x3904 | 15616x8768 | 136.9 |
| 15 (5x3) | **4864x2736** | 2736x4864 | 19456x10944 | 212.9 |
| 16 (4x4, 1:1 only) | **3904x3904** | — | 15616x15616 | 243.9 |

(That's the 16:9/9:16/1:1 spine — the Size Picker dropdown carries all 56 tested sizes across nine
aspects, with per-size verdicts in [docs/preset_sweep_report.md](docs/preset_sweep_report.md).)

Every other knob has exactly one correct value — copy the sizes above and leave these alone:

| setting | value | why |
|---|---|---|
| `max_tile` | 1024 | the trained envelope; lowering it shrinks tiles for VRAM headroom **without changing output size**, raising it re-invites the color collapse |
| `overlap` | 64 | the validated feather band (x4 in output px) |
| `degrade_sigma` | 0.0 | only raise it when decoding deliberately half-denoised latents |
| `latent_format` | `qwenimage` | for Qwen-family models (Krea 2, Qwen-Image); **`flux2` for FLUX.2 dev / Klein 4B / 9B**; `flux`/`sd3`/`sdxl` for those families |
| `seed` | anything | tile i noises with `seed + 101*(i+1)` |

Want more resolution? Render a bigger latent. There is deliberately no way to force *fewer* tiles
than planned — that would push tiles past the trained envelope, the exact failure this node exists
to prevent. The node prints its plan to the console on every run so you can see what it decided.

### FLUX.2 / Klein calibration (measured 2026-08-10)

Full 56-candidate sweep, **Klein 4B distilled** stage-1 → **PiD v1.5 flux2** decode, QA'd against
the `flux2-vae` twin on every rung: **chroma delta 9.2 min / 11.9 median / 18.9 max — all 56
inside the <20 chroma gate**, including the 208–244MP monster rungs; the whole sweep took ~50
GPU-minutes on the 5090. Two rungs flagged the secondary shadow green-bias gate (±3) and got a
seed-shifted retest: the **4:3 6-tile rung failed twice (−3.66, −5.15) and is culled** from the
dropdown — chroma healthy, but it doesn't ship until it measures clean, same discipline as the
v1.1.1 qwen culls; the 4:5 9-tile rung passed its retest (−0.66) and stays. **55 entries ship.**
Note the flux2 lane runs a few chroma points warmer than qwen at the biggest rungs — still
comfortably inside the gate, and the QA node prints both numbers on every run so drift is visible
immediately.

Stage-1 content reality check (same doctrine as the qwen ladder): Klein's native render envelope
is ~1–4MP. The decode stays faithful at any ladder size — what changes at monster rungs is the
stage-1 model's compositional coherence, so scenery holds up best up there; keep faces and hero
subjects at the ~2MP rungs.

Licensing: Klein 4B is Apache-2.0; **Klein 9B and FLUX.2-dev are non-commercial licenses**, and
the PiD weights are NVIDIA non-commercial regardless — the strictest license in your chain wins.

### Latent-Tiled PiD QA (vs VAE twin)

`IMAGE + IMAGE -> STRING`. Wire in the decode and the normal `VAEDecode` of the same latent.
Reports **midtone chroma delta** (the collapse detector — healthy decodes calibrate ~8–12 on PiD
v1.5 (10–15 on v1), collapsed ~30, flag > 20) and **shadow green-bias delta** (flag ±3). Run it
whenever you push a new resolution or model combo.

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/BennyDaBall930/ComfyUI-Latent-Tiled-PiD
```

Restart ComfyUI. Nodes are under `latent/pid`. Requires ComfyUI ≥ 0.28 (PiD in core) and the PiD
checkpoints (NVIDIA weights under the
[NVIDIA License](https://huggingface.co/nvidia/PixelDiT-1300M-1024px/blob/main/LICENSE) —
non-commercial research/evaluation use). No python dependencies beyond ComfyUI itself.
The **FLUX.2/Klein lane** additionally needs a ComfyUI new enough to ship FLUX.2 support
(`Flux2` latent format, `Flux2Scheduler`, CLIPLoader type `flux2` — late-2025 or newer; developed
and swept on **0.31.1**).

## Example workflows

[`example_workflows/krea2_turbo_sizepicker.json`](example_workflows/krea2_turbo_sizepicker.json) —
**the recommended one**: full Krea 2 Turbo pipeline with the Size Picker in front (one dropdown →
KSampler → tiled decode → master), VAE baseline and QA node wired in.

[`example_workflows/krea2_turbo_2mp_to_33mp.json`](example_workflows/krea2_turbo_2mp_to_33mp.json) —
the same pipeline with a stock `EmptySD3LatentImage` at fixed 1920x1088, for people who prefer
typing numbers.

[`example_workflows/flux2_klein_4b_tiled_pid.json`](example_workflows/flux2_klein_4b_tiled_pid.json)
— **FLUX.2 Klein 4B distilled** end-to-end: the flux2 Size Picker (width/height wired into
`Flux2Scheduler`, 4 steps, euler, CFG 1.0) → tiled PiD decode with the flux2 checkpoint → 33MP
master, VAE baseline + QA included. Models: `flux-2-klein-4b.safetensors`, `qwen_3_4b.safetensors`
(CLIPLoader type `flux2`), `flux2-vae.safetensors`, `pid_1.5_flux2_1024_to_4096_4step_bf16.safetensors`.

[`example_workflows/flux2_klein_9b_tiled_pid.json`](example_workflows/flux2_klein_9b_tiled_pid.json)
— the same graph on **Klein 9B distilled fp8** (`flux-2-klein-9b-fp8.safetensors` +
`qwen_3_8b_fp8mixed.safetensors`). Non-commercial license on the 9B weights. Validated clean
through the 59MP rungs — pick the 1/2/4/6-tile dropdown lines; for anything bigger use the 4B
workflow, which sweeps clean to 244MP.

Loaded in ComfyUI:

![the size-picker example loaded in ComfyUI — one dropdown replaces all resolution math](images/workflow_in_comfyui.jpg)

## Tested on

Works on my machine. Here is the machine, in its entirety — if your setup differs and something
breaks, this table is the first thing to check, not the issue tracker:

| component | exact version |
|---|---|
| ComfyUI | **0.31.1** (also validated on 0.30.1; built against `comfy.sample.sample_custom`, the pixel-space VAE, and core PiD support) |
| Python | 3.13.12 |
| PyTorch | 2.13.0+cu130 |
| stage-1 models | **Krea 2 Turbo** — `krea2_turbo_fp8_scaled.safetensors` + `qwen3vl_4b_fp8_scaled` text encoder (CLIPLoader type `krea2`), 8 steps euler/simple cfg 1.0 · **FLUX.2 Klein 4B distilled** — `flux-2-klein-4b.safetensors` + `qwen_3_4b.safetensors` (CLIPLoader type `flux2`), `Flux2Scheduler` 4 steps euler cfg 1.0 · **FLUX.2 Klein 9B distilled fp8** — `flux-2-klein-9b-fp8.safetensors` + `qwen_3_8b_fp8mixed.safetensors`, spot-validated PASS at the 9/33/59MP rungs, flagged at 111/137/244MP (keep 9B to ≤59MP outputs) |
| PiD checkpoints | **v1.5** — `pid_1.5_qwenimage_1024_to_4096_4step_bf16.safetensors` (qwen lane) and `pid_1.5_flux2_1024_to_4096_4step_bf16.safetensors` (FLUX.2 lane); v1 is deprecated upstream; on identical latents v1.5 roughly halved our midtone chroma delta and zeroed shadow green-bias |
| GPU | RTX 5090 32 GB, `--use-sage-attention`, cudaMallocAsync |
| verified outputs | qwen ladder: full 56-preset sweep to 244MP ([docs/preset_sweep_report.md](docs/preset_sweep_report.md)) · flux2 ladder: full 56-preset sweep to 244MP on Klein 4B ([docs/preset_sweep_report_flux2.md](docs/preset_sweep_report_flux2.md)) |
| test dates | 2026-08-04 → 2026-08-06 (qwen) · 2026-08-10 (FLUX.2/Klein) |

On the v1 checkpoint the biggest rungs (107MP+) showed a +3.5–3.9 shadow green-bias tone stretch,
disclosed in earlier releases; **on v1.5 those same rungs measure near zero** (−0.5 to +0.02). If
ComfyUI moves its sampling internals in some future version, that is when this pack needs a patch,
not an argument — open an issue **with your versions** and it gets fixed.

## Measurements, methodology, headless version

The full evidence set — reproducible demo, calibration numbers, the past-envelope collapse
characterization, and a headless driver that does the same thing over the ComfyUI API for pipeline
use — lives in the companion repo:
**[pid-tiled-decode](https://github.com/BennyDaBall930/pid-tiled-decode)**.

Also documented there: core ComfyUI `LatentCrop` silently corrupts 5-dim qwen/Wan-family latents
(indexes T/H as if they were H/W). This pack slices latents rank-agnostically and is not affected.

## Credit

- **NVIDIA** — PiD itself: Lu, Wu, Wu, Wang, Ling, Fidler, Ren
  ([nv-tlabs/PiD](https://github.com/nv-tlabs/PiD)). The decoder is theirs; this pack just refuses
  to leave its comfort zone.
- [Merserk/ComfyUI-PiD](https://github.com/Merserk/ComfyUI-PiD) — the image-space tiled upscaler
  design point, credited as prior art.
- Concept, direction and testing: **BennyDaBall930**.

## License & compliance

This repo's code and images (our own generations) are **MIT**. It ships **no NVIDIA model weights
and no NVIDIA code** — the PiD checkpoints come from NVIDIA under the
[NVIDIA License](https://huggingface.co/nvidia/PixelDiT-1300M-1024px/blob/main/LICENSE)
(non-commercial: research and evaluation use); read it before pointing this at anything commercial.
Independent project — not affiliated with, sponsored by, or endorsed by NVIDIA or Comfy Org.

Built & tested locally with care by BennyDaBall930.
