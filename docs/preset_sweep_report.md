# Full preset sweep report

2026-08-06, harbor-scene prompt, Krea 2 Turbo, RTX 5090. Every not-previously-tested
preset rendered end-to-end and gated vs its VAE twin (seam ratio, shadow green-bias,
midtone chroma). The nine ladder/portrait/square sizes were render-tested earlier
(see repo history). The three FAIL rows below were culled from the dropdown in v1.1.1.

total 47 | PASS 41 | MARGINAL 3 | FAIL 3 | RUN-FAILED 0

| preset | verdict | green | chroma | worst seam | s |
|---|---|---|---|---|---|
| 9:16 | render 576x1024 -> output 2304x4096 | 9 MP | 1 tile | PASS | 2.37 | 9.32 | 1.0 | 25.4 |
| 9:16 | render 768x1344 -> output 3072x5376 | 17 MP | 2 tiles | PASS | -0.02 | 9.53 | 0.747 | 25.9 |
| 9:16 | render 1088x1920 -> output 4352x7680 | 33 MP | 4 tiles | PASS | 1.52 | 8.96 | 1.041 | 37.8 |
| 9:16 | render 1944x3456 -> output 7776x13824 | 107 MP | 8 tiles | PASS | 1.66 | 14.11 | 1.029 | 109.2 |
| 9:16 | render 2192x3904 -> output 8768x15616 | 137 MP | 12 tiles | MARGINAL | 3.55 | 19.13 | 0.872 | 138.6 |
| 9:16 | render 2736x4864 -> output 10944x19456 | 213 MP | 15 tiles | FAIL | 3.04 | 21.11 | 0.89 | 226.6 |
| 4:3 | render 1024x768 -> output 4096x3072 | 13 MP | 1 tile | PASS | 0.79 | 15.63 | 1.0 | 20.7 |
| 4:3 | render 1360x1016 -> output 5440x4064 | 22 MP | 2 tiles | PASS | 2.94 | 11.72 | 0.732 | 30.9 |
| 4:3 | render 1984x1488 -> output 7936x5952 | 47 MP | 4 tiles | PASS | 1.2 | 12.89 | 0.78 | 52.4 |
| 4:3 | render 2640x1976 -> output 10560x7904 | 83 MP | 6 tiles | PASS | 2.4 | 12.62 | 0.984 | 88.5 |
| 4:3 | render 2944x2208 -> output 11776x8832 | 104 MP | 9 tiles | PASS | 1.24 | 13.62 | 0.853 | 103.1 |
| 4:3 | render 3904x2928 -> output 15616x11712 | 183 MP | 12 tiles | PASS | 2.12 | 16.78 | 1.015 | 199.1 |
| 3:4 | render 768x1024 -> output 3072x4096 | 13 MP | 1 tile | PASS | 2.65 | 10.74 | 1.0 | 20.6 |
| 3:4 | render 1024x1360 -> output 4096x5440 | 22 MP | 2 tiles | PASS | 0.58 | 13.05 | 0.702 | 30.3 |
| 3:4 | render 1488x1984 -> output 5952x7936 | 47 MP | 4 tiles | PASS | 2.81 | 11.7 | 0.864 | 53.3 |
| 3:4 | render 1984x2640 -> output 7936x10560 | 84 MP | 6 tiles | PASS | 1.5 | 14.85 | 0.946 | 92.9 |
| 3:4 | render 2208x2944 -> output 8832x11776 | 104 MP | 9 tiles | PASS | 2.57 | 13.69 | 0.932 | 110.9 |
| 3:4 | render 2928x3904 -> output 11712x15616 | 183 MP | 12 tiles | PASS | 2.8 | 18.87 | 0.971 | 207.7 |
| 3:2 | render 1024x680 -> output 4096x2720 | 11 MP | 1 tile | PASS | 0.1 | 14.54 | 1.0 | 20.8 |
| 3:2 | render 1536x1024 -> output 6144x4096 | 25 MP | 2 tiles | PASS | 2.57 | 11.24 | 0.958 | 36.1 |
| 3:2 | render 1984x1320 -> output 7936x5280 | 42 MP | 4 tiles | PASS | 0.83 | 14.15 | 0.909 | 52.0 |
| 3:2 | render 2944x1960 -> output 11776x7840 | 92 MP | 6 tiles | PASS | 2.46 | 14.12 | 0.921 | 99.3 |
| 3:2 | render 3904x2600 -> output 15616x10400 | 162 MP | 12 tiles | PASS | 1.75 | 16.04 | 0.943 | 177.9 |
| 3:2 | render 4416x2944 -> output 17664x11776 | 208 MP | 15 tiles | FAIL | 2.61 | 20.12 | 0.95 | 229.9 |
| 2:3 | render 680x1024 -> output 2720x4096 | 11 MP | 1 tile | PASS | 2.02 | 9.55 | 1.0 | 20.9 |
| 2:3 | render 1024x1536 -> output 4096x6144 | 25 MP | 2 tiles | PASS | 1.27 | 12.13 | 0.869 | 36.2 |
| 2:3 | render 1320x1976 -> output 5280x7904 | 42 MP | 4 tiles | PASS | 2.21 | 11.51 | 0.844 | 52.5 |
| 2:3 | render 1960x2936 -> output 7840x11744 | 92 MP | 6 tiles | PASS | 2.59 | 14.67 | 0.811 | 100.0 |
| 2:3 | render 2600x3896 -> output 10400x15584 | 162 MP | 12 tiles | PASS | 2.17 | 15.46 | 0.852 | 170.1 |
| 2:3 | render 2944x4416 -> output 11776x17664 | 208 MP | 15 tiles | FAIL | 2.39 | 20.91 | 0.925 | 217.1 |
| 4:5 | render 816x1024 -> output 3264x4096 | 13 MP | 1 tile | PASS | 2.72 | 11.34 | 1.0 | 20.6 |
| 4:5 | render 1024x1280 -> output 4096x5120 | 21 MP | 2 tiles | PASS | 0.79 | 12.09 | 0.924 | 26.4 |
| 4:5 | render 1584x1976 -> output 6336x7904 | 50 MP | 4 tiles | PASS | 2.65 | 11.36 | 0.903 | 52.7 |
| 4:5 | render 1984x2480 -> output 7936x9920 | 79 MP | 6 tiles | PASS | 1.35 | 13.19 | 0.989 | 82.6 |
| 4:5 | render 2352x2936 -> output 9408x11744 | 110 MP | 9 tiles | PASS | 2.46 | 13.73 | 0.865 | 109.7 |
| 4:5 | render 2944x3680 -> output 11776x14720 | 173 MP | 12 tiles | PASS | 1.82 | 16.82 | 0.866 | 179.5 |
| 4:5 | render 3120x3896 -> output 12480x15584 | 194 MP | 16 tiles | MARGINAL | 3.01 | 18.05 | 0.855 | 201.6 |
| 5:4 | render 1024x816 -> output 4096x3264 | 13 MP | 1 tile | PASS | 1.09 | 15.39 | 1.0 | 20.6 |
| 5:4 | render 1280x1024 -> output 5120x4096 | 21 MP | 2 tiles | MARGINAL | 3.01 | 9.74 | 1.03 | 26.3 |
| 5:4 | render 1984x1584 -> output 7936x6336 | 50 MP | 4 tiles | PASS | 1.54 | 13.6 | 0.938 | 52.5 |
| 5:4 | render 2480x1984 -> output 9920x7936 | 79 MP | 6 tiles | PASS | 2.65 | 16.63 | 0.88 | 82.7 |
| 5:4 | render 2944x2352 -> output 11776x9408 | 111 MP | 9 tiles | PASS | 1.08 | 17.22 | 0.905 | 109.4 |
| 5:4 | render 3680x2944 -> output 14720x11776 | 173 MP | 12 tiles | PASS | 2.25 | 17.54 | 0.857 | 179.4 |
| 5:4 | render 3904x3120 -> output 15616x12480 | 195 MP | 16 tiles | PASS | 1.68 | 18.21 | 0.865 | 197.4 |
| 1:1 | render 1024x1024 -> output 4096x4096 | 17 MP | 1 tile | PASS | 2.25 | 13.52 | 1.0 | 25.2 |
| 1:1 | render 1984x1984 -> output 7936x7936 | 63 MP | 4 tiles | PASS | 2.67 | 14.71 | 1.093 | 67.0 |
| 1:1 | render 2944x2944 -> output 11776x11776 | 139 MP | 9 tiles | PASS | 1.92 | 17.55 | 0.944 | 143.2 |

