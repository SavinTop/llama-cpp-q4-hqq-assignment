# Task 2 — Q4_HQQ versus Q4_0

## Model size

The final ordinary Q4_HQQ quantization kept tied token embeddings fully quantized.

| Format | File size | Relative size |
|---|---:|---:|
| Q4_0 | 1,917,190,592 bytes | baseline |
| Q4_HQQ | 2,016,397,760 bytes | +5.17% |

Final Q4_HQQ SHA-256:

```text
FD44187FD1BA53C383F3D7B1BD1F0D5C2FB5EA48D23526170814DE563121903E
```

## Controlled Windows performance

Same Llama 3.2 3B source, ten CPU threads, pp512/tg128 and seven repetitions.

| Backend | Quant | pp512 median | tg128 median |
|---|---|---:|---:|
| CPU | Q4_0 | 78.9888 tokens/s | 9.81128 tokens/s |
| CPU | Q4_HQQ | 44.7974 tokens/s | 8.95402 tokens/s |
| Vulkan0 | Q4_0 | 2737.24 tokens/s | 78.7885 tokens/s |
| Vulkan0 | Q4_HQQ | 910.791 tokens/s | 66.8413 tokens/s |

Raw benchmark output is stored under `benchmarks/`.

## Output quality

Both formats produced coherent responses to:

- `What is bitcoin?`
- `Write a Python function to reverse a list.`

The smoke outputs are stored under `quality/`.

## Perplexity bonus

A bounded four-chunk Wikitext-2 run used identical CPU settings.

| Format | Perplexity | Wall time |
|---|---:|---:|
| Q4_0 | 9.4473 +/- 0.82340 | 33.099 s |
| Q4_HQQ | 9.3079 +/- 0.80037 | 107.346 s |

The uncertainty intervals overlap. This bounded run detected no Q4_HQQ regression but does not establish an improvement.

## Interpretation

Q4_HQQ generation speed remains closer to Q4_0 than prompt-processing speed. Q4_0 benefits from heavily optimized architecture-specific matrix paths that are not yet available for Q4_HQQ. The Q4_HQQ format uses 5.0 bits per weight at block level and therefore has a modest size overhead relative to Q4_0.
