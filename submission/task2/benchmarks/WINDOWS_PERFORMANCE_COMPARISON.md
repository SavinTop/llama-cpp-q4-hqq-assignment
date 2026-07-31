# Median comparison

All headline Windows baselines/finals use identical model files, `pp512`,
`tg128`, ten CPU threads and seven repetitions (`-r 7`). Values are tokens/s.

## CPU

| Metric | Q4_0 | Q4_HQQ before | Q4_HQQ after | HQQ/Q4_0 |
|---|---:|---:|---:|---:|
| pp512 | 78.9888 | 44.7974 | 44.7974* | 56.71% |
| tg128 | 9.81128 | 8.95402 | 8.95402* | 91.26% |

`*` The retained diff is Vulkan-shader-only. No CPU source/binary behavior
changed, so the fresh baseline is the source-identical final CPU result; it is
not presented as a second measurement.

CPU raw Q4_HQQ pp512:
`44.6925, 44.9644, 44.3665, 44.9048, 44.1756, 44.7974, 44.8545`

CPU raw Q4_HQQ tg128:
`8.79177, 8.82929, 8.91475, 9.02799, 8.95530, 9.09674, 8.95402`

## NVIDIA Vulkan0

| Metric | Q4_0 before | Q4_0 after | change | Q4_HQQ before | Q4_HQQ after | change |
|---|---:|---:|---:|---:|---:|---:|
| pp512 | 2716.20 | 2737.24 | +0.77% | 883.395 | 910.791 | +3.10% |
| tg128 | 78.7316 | 78.7885 | +0.07% | 66.9241 | 66.8413 | -0.12% |

Final NVIDIA Q4_HQQ/Q4_0 pp512 ratio: `910.791 / 2737.24 = 33.27%`.

NVIDIA Q4_HQQ pp512 before:
`885.128, 887.727, 883.819, 878.884, 882.350, 883.395, 882.841`

NVIDIA Q4_HQQ pp512 after:
`911.396, 912.692, 909.969, 912.197, 910.018, 910.791, 909.295`

NVIDIA Q4_HQQ tg128 before:
`66.6710, 66.9475, 66.9508, 67.0115, 66.7874, 66.6978, 66.9241`

NVIDIA Q4_HQQ tg128 after:
`67.0082, 66.8413, 66.9168, 66.8298, 66.6936, 66.8851, 66.7505`

## Intel Vulkan1 post-change smoke benchmark

| Metric | Q4_0 | Q4_HQQ |
|---|---:|---:|
| pp512 | 245.957 | 89.6854 |
| tg128 | 9.02838 | 8.28287 |
