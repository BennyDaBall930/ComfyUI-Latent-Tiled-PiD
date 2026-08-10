# FLUX.2 / Klein preset sweep — Latent-Tiled PiD v1.3.0

Date: 2026-08-10  
Stage-1: **FLUX.2 Klein 4B distilled** (`flux-2-klein-4b.safetensors`, 4 steps, euler, CFG 1.0 via Flux2Scheduler/CFGGuider)  
Decoder: **PiD v1.5 flux2** (`pid_1.5_flux2_1024_to_4096_4step_bf16.safetensors`), max_tile 1024, overlap 64, seed 508040300  
QA: midtone chroma delta vs `flux2-vae.safetensors` twin (flag > 20.0), shadow green-bias delta (flag ±3.0)

Stage-1 content note: Klein's native render envelope is ~1–4MP. The monster rungs stress the DECODER's fidelity to whatever the latent contains — at very large stage-1 sizes, composition quality is the stage-1 model's affair, exactly as with the qwen-family ladder (scenery holds up best).

| preset | chroma Δ | green Δ | wall s | verdict |
|---|---|---|---|---|
| 16:9 | render 1024x576 -> output 4096x2304 | 9 MP | 1 tile | 12.29 | -0.38 | 24 | PASS |
| 16:9 | render 1344x768 -> output 5376x3072 | 17 MP | 2 tiles | 10.93 | -0.68 | 8 | PASS |
| 16:9 | render 1920x1088 -> output 7680x4352 | 33 MP | 4 tiles | 10.95 | -0.37 | 16 | PASS |
| 16:9 | render 2560x1440 -> output 10240x5760 | 59 MP | 6 tiles | 11.37 | +0.25 | 32 | PASS |
| 16:9 | render 3520x1968 -> output 14080x7872 | 111 MP | 8 tiles | 12.61 | +1.07 | 64 | PASS |
| 16:9 | render 3904x2192 -> output 15616x8768 | 137 MP | 12 tiles | 16.70 | +0.32 | 80 | PASS |
| 16:9 | render 4864x2736 -> output 19456x10944 | 213 MP | 15 tiles | 18.76 | +0.49 | 148 | PASS |
| 9:16 | render 576x1024 -> output 2304x4096 | 9 MP | 1 tile | 12.63 | -0.98 | 12 | PASS |
| 9:16 | render 768x1344 -> output 3072x5376 | 17 MP | 2 tiles | 9.82 | -1.61 | 8 | PASS |
| 9:16 | render 1088x1920 -> output 4352x7680 | 33 MP | 4 tiles | 9.68 | +0.55 | 16 | PASS |
| 9:16 | render 1440x2560 -> output 5760x10240 | 59 MP | 6 tiles | 10.76 | +0.47 | 32 | PASS |
| 9:16 | render 1984x3520 -> output 7936x14080 | 112 MP | 8 tiles | 16.65 | -1.10 | 64 | PASS |
| 9:16 | render 2192x3904 -> output 8768x15616 | 137 MP | 12 tiles | 14.84 | +0.14 | 80 | PASS |
| 9:16 | render 2736x4864 -> output 10944x19456 | 213 MP | 15 tiles | 16.15 | -0.67 | 144 | PASS |
| 4:3 | render 1024x768 -> output 4096x3072 | 13 MP | 1 tile | 9.90 | -0.09 | 8 | PASS |
| 4:3 | render 1360x1008 -> output 5440x4032 | 22 MP | 2 tiles | 11.34 | -0.44 | 12 | PASS |
| 4:3 | render 1984x1488 -> output 7936x5952 | 47 MP | 4 tiles | 9.41 | +1.20 | 28 | PASS |
| 4:3 | render 2640x1968 -> output 10560x7872 | 83 MP | 6 tiles | 11.84 | -3.66 | 52 | FLAG |
| 4:3 | render 2944x2208 -> output 11776x8832 | 104 MP | 9 tiles | 17.41 | +0.15 | 56 | PASS |
| 4:3 | render 3904x2928 -> output 15616x11712 | 183 MP | 12 tiles | 18.34 | +0.04 | 116 | PASS |
| 3:4 | render 768x1024 -> output 3072x4096 | 13 MP | 1 tile | 9.90 | +0.71 | 8 | PASS |
| 3:4 | render 1024x1360 -> output 4096x5440 | 22 MP | 2 tiles | 10.93 | -0.12 | 12 | PASS |
| 3:4 | render 1488x1984 -> output 5952x7936 | 47 MP | 4 tiles | 10.72 | -0.41 | 28 | PASS |
| 3:4 | render 1984x2640 -> output 7936x10560 | 84 MP | 6 tiles | 11.78 | -1.50 | 48 | PASS |
| 3:4 | render 2208x2944 -> output 8832x11776 | 104 MP | 9 tiles | 17.89 | +0.70 | 56 | PASS |
| 3:4 | render 2928x3904 -> output 11712x15616 | 183 MP | 12 tiles | 16.80 | -0.63 | 116 | PASS |
| 3:2 | render 1024x672 -> output 4096x2688 | 11 MP | 1 tile | 12.24 | +0.05 | 8 | PASS |
| 3:2 | render 1536x1024 -> output 6144x4096 | 25 MP | 2 tiles | 10.15 | -0.20 | 16 | PASS |
| 3:2 | render 1984x1312 -> output 7936x5248 | 42 MP | 4 tiles | 10.38 | -0.99 | 24 | PASS |
| 3:2 | render 2944x1952 -> output 11776x7808 | 92 MP | 6 tiles | 13.32 | -1.62 | 56 | PASS |
| 3:2 | render 3904x2592 -> output 15616x10368 | 162 MP | 12 tiles | 18.86 | -0.39 | 96 | PASS |
| 3:2 | render 4416x2944 -> output 17664x11776 | 208 MP | 15 tiles | 18.77 | -0.50 | 144 | PASS |
| 2:3 | render 672x1024 -> output 2688x4096 | 11 MP | 1 tile | 11.76 | -0.20 | 8 | PASS |
| 2:3 | render 1024x1536 -> output 4096x6144 | 25 MP | 2 tiles | 10.74 | -2.01 | 16 | PASS |
| 2:3 | render 1312x1968 -> output 5248x7872 | 41 MP | 4 tiles | 10.21 | -1.47 | 24 | PASS |
| 2:3 | render 1952x2928 -> output 7808x11712 | 91 MP | 6 tiles | 11.87 | -1.63 | 52 | PASS |
| 2:3 | render 2592x3888 -> output 10368x15552 | 161 MP | 12 tiles | 15.60 | -0.23 | 104 | PASS |
| 2:3 | render 2944x4416 -> output 11776x17664 | 208 MP | 15 tiles | 15.92 | +0.20 | 136 | PASS |
| 4:5 | render 816x1024 -> output 3264x4096 | 13 MP | 1 tile | 9.37 | +1.47 | 8 | PASS |
| 4:5 | render 1024x1280 -> output 4096x5120 | 21 MP | 2 tiles | 11.13 | -0.47 | 12 | PASS |
| 4:5 | render 1584x1968 -> output 6336x7872 | 50 MP | 4 tiles | 10.77 | +0.29 | 28 | PASS |
| 4:5 | render 1984x2480 -> output 7936x9920 | 79 MP | 6 tiles | 10.85 | -0.38 | 44 | PASS |
| 4:5 | render 2352x2928 -> output 9408x11712 | 110 MP | 9 tiles | 18.13 | -4.03 | 64 | FLAG |
| 4:5 | render 2944x3680 -> output 11776x14720 | 173 MP | 12 tiles | 17.21 | -0.19 | 108 | PASS |
| 4:5 | render 3120x3888 -> output 12480x15552 | 194 MP | 16 tiles | 18.47 | -0.31 | 120 | PASS |
| 5:4 | render 1024x816 -> output 4096x3264 | 13 MP | 1 tile | 10.15 | +0.16 | 8 | PASS |
| 5:4 | render 1280x1024 -> output 5120x4096 | 21 MP | 2 tiles | 12.38 | -0.67 | 12 | PASS |
| 5:4 | render 1984x1584 -> output 7936x6336 | 50 MP | 4 tiles | 10.74 | +0.05 | 28 | PASS |
| 5:4 | render 2480x1984 -> output 9920x7936 | 79 MP | 6 tiles | 11.65 | -0.57 | 44 | PASS |
| 5:4 | render 2944x2352 -> output 11776x9408 | 111 MP | 9 tiles | 16.15 | +0.02 | 63 | PASS |
| 5:4 | render 3680x2944 -> output 14720x11776 | 173 MP | 12 tiles | 18.16 | -0.75 | 108 | PASS |
| 5:4 | render 3904x3120 -> output 15616x12480 | 195 MP | 16 tiles | 15.77 | +0.17 | 116 | PASS |
| 1:1 | render 1024x1024 -> output 4096x4096 | 17 MP | 1 tile | 9.37 | +0.38 | 12 | PASS |
| 1:1 | render 1984x1984 -> output 7936x7936 | 63 MP | 4 tiles | 9.18 | -0.59 | 36 | PASS |
| 1:1 | render 2944x2944 -> output 11776x11776 | 139 MP | 9 tiles | 17.22 | -1.46 | 84 | PASS |
| 1:1 | render 3904x3904 -> output 15616x15616 | 244 MP | 16 tiles | 17.38 | +1.10 | 172 | PASS |

**56 presets measured** — chroma min/median/max: 9.18 / 11.87 / 18.86; pass: 54, flag: 2.

## Seed-shift retest & cull decision

Both green-bias FLAG rows above were rerun with seed 508040301 (same rung, different noise):

| preset | first run | retest | decision |
|---|---|---|---|
| 4:3 \| render 2640x1968 (6 tiles, 83 MP) | green −3.66 | green **−5.15** (chroma 13.18) | **CULLED** — systematic shadow drift on Klein 4B; chroma healthy both runs; restoration candidate if a future flux2 PiD checkpoint fixes it |
| 4:5 \| render 2352x2928 (9 tiles, 110 MP) | green −4.03 | green **−0.66** (chroma 16.07) | **KEPT** — first flag was content/seed noise of the shadow metric |

Shipped flux2 ladder: **55 presets** (56 candidates − 1 cull).
