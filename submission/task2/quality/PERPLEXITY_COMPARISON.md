# Stage 07 comparison

## AVX2 performance

The Q4_HQQ benchmark used the same model and settings as Stage 06.

| Test | Stage 06 scalar | Stage 07 AVX2 | Speedup |
|---|---:|---:|---:|
| pp512 | 4.67 +/- 0.13 tok/s | 19.98 +/- 0.24 tok/s | 4.28x |
| tg128 | 3.48 +/- 0.02 tok/s | 6.15 +/- 0.23 tok/s | 1.77x |

The Stage 07 values are the arithmetic mean and sample standard deviation from five repetitions.

## Perplexity

Both runs used the first four 512-token chunks of the same Wikitext-2 raw test corpus, CPU-only execution, 10 threads, batch 512, ubatch 512, no warmup, and identical model/runtime settings.

| Quant | Final PPL | Wall time |
|---|---:|---:|
| Q4_0 | 9.4473 +/- 0.82340 | 33.099 s |
| Q4_HQQ | 9.3079 +/- 0.80037 | 107.346 s |

Q4_HQQ is 0.1394 lower in this limited run, a 1.48% difference. The uncertainty intervals overlap substantially, so the result shows no detected PPL regression in this sample but does not establish that Q4_HQQ is better.
