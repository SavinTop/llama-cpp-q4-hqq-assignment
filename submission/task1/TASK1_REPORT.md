# Task 1 — GGUF Conversion, Q4_0 Quantization and CPU Benchmark

## Result

`meta-llama/Llama-3.2-3B-Instruct` was converted to F16 GGUF, quantized to `Q4_0`, validated with two inference prompts, and benchmarked on CPU.

## Environment

- OS: Windows 11 Pro 64-bit (`10.0.26200`)
- CPU: Intel Core i7-12650H — 10 cores, 16 logical processors
- Benchmark threads: 10
- Installed RAM: 31.73 GiB
- Backend: CPU
- Build: Release, MSVC `19.51.36248.0`
- `GGML_NATIVE=ON`
- `GGML_CUDA=OFF`
- `GGML_VULKAN=OFF`
- `llama.cpp` build: `10121`
- Commit: `555881ebc8b0fc0402b30e09258a32a7bfd13c52`

## Model

- Source: `meta-llama/Llama-3.2-3B-Instruct`
- Source URL: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
- Output format: GGUF `Q4_0`
- Parameters: `3,212,749,888`
- Output file size: `1.786 GiB`
- SHA-256: `CAD0FC6144407850FFC131C4CCAA05C83D675735BB108090A839D145AC6AA767`

The conversion log confirms successful export to F16 GGUF. The quantization log confirms conversion of 255 tensors from an approximately `6128.17 MiB` F16 model to a `1820.90 MiB` Q4_0 representation.

## Inference validation

The Q4_0 model was successfully loaded and produced coherent responses for:

1. `What is bitcoin?`
2. `Write a Python function to reverse a list.`

Generation was intentionally capped at 128 tokens for each validation run.

## Benchmark methodology

Benchmarks were executed with `llama-bench` on the CPU backend using 10 threads.

- First-token compute test: synthetic 32-token prompt followed by one generated token, 10 repetitions.
- Single-token prompt evaluation: one synthetic prompt token, 10 repetitions.
- Prompt processing: 512 prompt tokens, 5 repetitions.
- Token generation: 128 generated tokens, 5 repetitions.
- RAM: five independent `llama-bench` processes running the `pp512 + tg128` workload.

The first-token result below is a model-compute measurement, not end-to-end application latency. It does not include model loading or terminal/UI overhead.

## Benchmark results

| Metric | Result |
|---|---:|
| First-token compute latency — 32 prompt tokens + 1 generated token | `473.94 ± 23.77 ms` |
| Single-token prompt evaluation (`pp1`) | `101.09 ± 8.80 ms` |
| Prompt processing (`pp512`) | `76.54 ± 3.11 tokens/s` |
| Token generation (`tg128`) | `10.59 ± 0.03 tokens/s` |
| Average generation time | `94.43 ms/token` |
| Average peak working set | `3534.18 MiB` (`3.451 GiB`) |
| Maximum peak working set | `3534.24 MiB` |
| Average peak private memory | `1949.51 MiB` |

## Reproduction

CPU build:

```powershell
cmake -S . -B build `
    -DGGML_NATIVE=ON `
    -DGGML_CUDA=OFF `
    -DGGML_VULKAN=OFF

cmake --build build --config Release -j
```

The scripts included in `scripts/` reproduce conversion, quantization, inference validation, benchmarks, and RAM measurements.

## Included artifacts

- `scripts/Convert-Model.ps1`
- `scripts/Quantize-Model.ps1`
- `scripts/Run-Task1FinalTests.ps1`
- `logs/conversion.log`
- `logs/quantization-q4_0.log`
- `logs/inference-bitcoin.log`
- `logs/inference-python.log`
- `logs/bench-pp1.json`
- `logs/bench-pg32-1.json`
- `logs/bench-performance.json`
- `logs/memory-results.csv`
- `logs/memory-summary.txt`
- `logs/environment.txt`

The original UTF-16 PowerShell logs are preserved under `logs/raw/`. The main conversion and quantization logs were normalized to UTF-8 and had only the PowerShell `NativeCommandError` wrapper removed; the underlying tool output was preserved.
