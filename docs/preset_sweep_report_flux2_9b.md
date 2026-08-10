# FLUX.2 / Klein preset sweep — Latent-Tiled PiD v1.3.0

Date: 2026-08-10  
Stage-1: **FLUX.2 Klein 4B distilled** (`flux-2-klein-9b-fp8.safetensors`, 4 steps, euler, CFG 1.0 via Flux2Scheduler/CFGGuider)  
Decoder: **PiD v1.5 flux2** (`pid_1.5_flux2_1024_to_4096_4step_bf16.safetensors`), max_tile 1024, overlap 64, seed 508040300  
QA: midtone chroma delta vs `flux2-vae.safetensors` twin (flag > 20.0), shadow green-bias delta (flag ±3.0)

Stage-1 content note: Klein's native render envelope is ~1–4MP. The monster rungs stress the DECODER's fidelity to whatever the latent contains — at very large stage-1 sizes, composition quality is the stage-1 model's affair, exactly as with the qwen-family ladder (scenery holds up best).

| preset | chroma Δ | green Δ | wall s | verdict |
|---|---|---|---|---|
| 16:9 | render 1024x576 -> output 4096x2304 | 9 MP | 1 tile | 13.89 | +1.76 | 12 | PASS |
| 16:9 | render 1920x1088 -> output 7680x4352 | 33 MP | 4 tiles | 10.85 | -0.31 | 16 | PASS |
| 16:9 | render 2560x1440 -> output 10240x5760 | 59 MP | 6 tiles | 10.22 | +0.38 | 32 | PASS |
| 16:9 | render 3520x1968 -> output 14080x7872 | 111 MP | 8 tiles | 17.63 | -7.65 | 64 | FLAG |
| 16:9 | render 3904x2192 -> output 15616x8768 | 137 MP | 12 tiles | 21.35 | -2.75 | 84 | FLAG |
| 1:1 | render 3904x3904 -> output 15616x15616 | 244 MP | 16 tiles | 28.48 | -7.97 | 168 | FLAG |

**6 presets measured** — chroma min/median/max: 10.22 / 17.63 / 28.48; pass: 3, flag: 3.
