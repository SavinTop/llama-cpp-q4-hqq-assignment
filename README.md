# QVAC Foundation Engineer Assignment

**Candidate:** Vladislav Savinov
**Repository base:** `555881ebc8b0fc0402b30e09258a32a7bfd13c52`
**Working branch:** `task2-q4-hqq`
**Primary platform:** Windows 11, x86-64
**Additional validation:** Ubuntu WSL2 and macOS on Apple M1

---

## 1. Environment and Setup

### 1.1 Primary Windows environment

| Item | Value |
|---|---|
| Operating system | Windows 11, build `10.0.26200.8875` |
| CPU | Intel Core i7-12650H, 16 logical processors exposed |
| Compiler | MSVC `19.51.36248` x64 |
| CMake | `4.4.0` |
| Generator | Ninja |
| CPU features | AVX2, FMA, F16C through `GGML_NATIVE=ON` |
| Vulkan SDK | `1.4.357.0` |
| Vulkan0 | NVIDIA GeForce RTX 3050 Ti Laptop GPU, 3962 MiB |
| Vulkan1 | Intel UHD Graphics |

### 1.2 Additional platforms

| Platform | Configuration | Result |
|---|---|---|
| Ubuntu 26.04 WSL2 | GCC 15.2, Ninja, native CPU build | Full build and focused tests passed |
| macOS, Apple M1 | ARM64 CPU with Accelerate | Q4_HQQ inference and benchmarks passed |
| macOS, Apple M1 | Vulkan through MoltenVK 1.4.1 | Q4_HQQ backend tests, full offload, and benchmarks passed |
| macOS, Apple M1 | Native Metal | Multimodal projector selection passed |

### 1.3 CPU build

```powershell
cmake -S . -B build-cpu `
  -DGGML_NATIVE=ON `
  -DGGML_CUDA=OFF `
  -DGGML_VULKAN=OFF `
  -DLLAMA_BUILD_TESTS=ON `
  -DLLAMA_BUILD_TOOLS=ON

cmake --build build-cpu --config Release --parallel
```

### 1.4 Vulkan build

```powershell
cmake -S . -B build-vulkan `
  -DGGML_NATIVE=ON `
  -DGGML_CUDA=OFF `
  -DGGML_VULKAN=ON `
  -DLLAMA_BUILD_TESTS=ON `
  -DLLAMA_BUILD_TOOLS=ON

cmake --build build-vulkan --config Release --parallel
```

### 1.5 Linux or WSL2 CPU build

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build

cmake -S . -B build-linux -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DGGML_VULKAN=OFF

cmake --build build-linux --parallel
```

### 1.6 macOS Apple Silicon Vulkan build

```bash
source "$HOME/VulkanSDK/<version>/setup-env.sh"
vulkaninfo --summary

cmake -S . -B build-mac-vulkan -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DGGML_METAL=OFF \
  -DGGML_VULKAN=ON

cmake --build build-mac-vulkan --parallel \
  --target llama-cli llama-bench llama-mtmd-cli \
           test-backend-ops test-arg-parser test-mtmd-c-api
```

---

## 2. Task 1: Quantize and Run a Model

### 2.1 Model

- Source model: `meta-llama/Llama-3.2-3B-Instruct`
- Converted format: F16 GGUF
- Quantized format: Q4_0
- Q4_0 file size: `1,917,190,592` bytes (`1828.38` MiB)

### 2.2 Download and conversion

```powershell
hf download meta-llama/Llama-3.2-3B-Instruct `
  --local-dir .\models\Llama-3.2-3B-Instruct

python .\convert_hf_to_gguf.py `
  .\models\Llama-3.2-3B-Instruct `
  --outfile .\models\llama-3.2-3b-instruct-f16.gguf `
  --outtype f16
```

The conversion uses the repository's GGUF conversion script and preserves the model metadata required by `llama.cpp`.

### 2.3 Q4_0 quantization

```powershell
.\build-cpu\bin\Release\llama-quantize.exe `
  .\models\llama-3.2-3b-instruct-f16.gguf `
  .\models\llama-3.2-3b-instruct-q4_0.gguf `
  Q4_0
```

### 2.4 Inference commands

```powershell
.\build-cpu\bin\Release\llama-cli.exe `
  -m .\models\llama-3.2-3b-instruct-q4_0.gguf `
  -p "What is bitcoin?" `
  -n 128 -t 10 -ngl 0 --seed 42 --single-turn

.\build-cpu\bin\Release\llama-cli.exe `
  -m .\models\llama-3.2-3b-instruct-q4_0.gguf `
  -p "Write a Python function to reverse a list." `
  -n 128 -t 10 -ngl 0 --seed 42 --single-turn
```

### 2.5 Output check

| Prompt | Result |
|---|---|
| `What is bitcoin?` | Coherent explanation of Bitcoin, decentralization, blockchain, and cryptographic verification |
| `Write a Python function to reverse a list.` | Valid Python implementation with a short explanation and example |

No repeated garbage, invalid token sequences, or obvious model corruption was observed.

### 2.6 Measurements

The benchmark used ten CPU threads. Throughput values are arithmetic mean and sample standard deviation.

| Metric | Result |
|---|---:|
| Prompt processing, `pp512` | `77.71 +/- 4.26` tokens/s |
| Token generation, `tg128` | `8.92 +/- 0.26` tokens/s |
| Single-token evaluation, `pp1` | `8.30 +/- 0.44` tokens/s |
| Single-token evaluation latency | `120.5 ms` |
| Peak working set, five-run average | approximately `3534 MiB` |
| Peak private memory, five-run average | approximately `1949 MiB` |
| End-to-end elapsed time | approximately `29.9-34.6 s` per run |

The latency value is the reciprocal of the measured single-token evaluation throughput. It does not include model loading time.

### 2.7 Task 1 artifacts

The submission includes the conversion and quantization commands, quantizer output, prompt output logs, benchmark results, and memory-measurement results.

---

## 3. Task 2: Q4_HQQ Quantization Format

### 3.1 Block layout

```c
#define QK4_HQQ 32

typedef struct {
    ggml_half scale;
    ggml_half zero;
    uint8_t qs[QK4_HQQ / 2];
} block_q4_hqq;
```

The layout is 20 bytes for 32 values:

```text
2-byte scale + 2-byte zero + 16 packed bytes = 20 bytes
20 * 8 / 32 = 5.0 bits per weight
```

Static assertions verify the block size and storage cost.

### 3.2 Quantization and dequantization

For each block:

```text
scale = 15.0 / (max - min)
zero  = -min * scale
q     = clamp(round(w * scale + zero), 0, 15)
```

Dequantization:

```text
w = (q - zero) / scale
```

Two values are packed into each byte. The lower nibbles represent values 0-15 and the upper nibbles represent values 16-31.

### 3.3 Finite metadata and constant blocks

`scale` and `zero` are stored as FP16. Direct conversion of ideal FP32 metadata can overflow for very small ranges or large offsets. The quantizer therefore stores FP16 metadata first and uses the stored values when generating the codes.

For non-constant blocks, the scale is reduced until both stored metadata values are finite. For constant blocks, a finite positive scale is selected and all codes are set to zero. The decoded value is calculated from the stored FP16 metadata.

A raw all-zero 20-byte block is reserved for zero-initialized KV-cache memory. It decodes to zero in CPU and Vulkan cache operations. Persisted quantized tensors still reject a zero scale during validation.

### 3.4 Q4_HQQ x Q8_0 dot product

For Q4_HQQ weights and Q8_0 activations:

```text
result += d8 / scale4 *
          (sum(q4_code * q8_code) - zero4 * sum(q8_code))
```

The zero point is converted from FP16 to FP32 and remains floating point. It is not rounded or cast to an integer.

### 3.5 CPU implementation

The CPU backend contains:

- scalar quantization and dequantization;
- scalar Q4_HQQ x Q8_0 dot product;
- portable fallback for unsupported architectures;
- AVX2 dot-product implementation;
- two-row AVX2 execution for prompt processing;
- single-row fallback for odd matrix dimensions and non-AVX2 builds.

The two-row path reuses unpacked weight rows and Q8 sums across a 2x2 output tile. The active prompt-processing trace confirms `nrc=2` dispatch on AVX2.

### 3.6 GGUF, model loading, and quantizer integration

The assignment listed integration points based on an older file layout. In the tested revision, the corresponding code is distributed as follows:

| Assignment area | Current integration point |
|---|---|
| Block definition | `ggml/src/ggml-common.h` |
| Public GGML type | `ggml/include/ggml.h` |
| Type traits and quantization dispatch | `ggml/src/ggml.c`, CPU backend traits |
| Quantization declarations and implementation | `ggml/src/ggml-quants.h`, `ggml/src/ggml-quants.c` |
| CPU dot product | `ggml/src/ggml-cpu/quants.h`, `ggml/src/ggml-cpu/quants.c` |
| Model type and loading | `include/llama.h`, `src/llama-model-loader.cpp` |
| Quantization policy | `src/llama-quant.cpp` |
| CLI quantization option | `tools/quantize/quantize.cpp` |
| Python GGUF metadata | `gguf-py/gguf/constants.py` |

The implementation registers Q4_HQQ for GGUF serialization, tensor sizing, quantization selection, model loading, CPU dispatch, and backend operation checks.

### 3.7 Quantization command

```powershell
.\build-cpu\bin\Release\llama-quantize.exe `
  .\models\llama-3.2-3b-instruct-f16.gguf `
  .\models\llama-3.2-3b-instruct-q4_hqq.gguf `
  Q4_HQQ
```

The quantized model can be reproduced from the F16 GGUF with this command. The generated model is not required in the source repository; revision-specific hashes are recorded in the validation manifests.

### 3.8 CPU inference

```powershell
.\build-cpu\bin\Release\llama-cli.exe `
  -m .\models\llama-3.2-3b-instruct-q4_hqq.gguf `
  -p "What is bitcoin?" `
  -n 128 -t 10 -ngl 0 --seed 42 --single-turn
```

Both requested prompts produced coherent output on the CPU backend.

---

## 4. Q4_HQQ KV Cache

### 4.1 CLI support

The standard cache options accept Q4_HQQ:

```text
--cache-type-k q4_hqq
--cache-type-v q4_hqq
```

Example:

```powershell
.\build-cpu\bin\Release\llama-cli.exe `
  -m .\models\llama-3.2-3b-instruct-q4_hqq.gguf `
  -p "What is bitcoin?" `
  -n 128 -c 32768 -t 10 -ngl 0 --seed 42 `
  --flash-attn on `
  --cache-type-k q4_hqq `
  --cache-type-v q4_hqq
```

Quantized V cache requires Flash Attention. Q4_HQQ K with F16 V is also supported without Flash Attention.

### 4.2 Cache allocation comparison

The repeated memory test used a 32768-token context, ten CPU threads, seed 42, and three fresh processes per cache type.

| Metric | F16 K/V | Q4_HQQ K/V | Difference |
|---|---:|---:|---:|
| Allocated KV cache | `3584 MiB` | `1120 MiB` | `3.20x` smaller |
| K cache | `1792 MiB` | `560 MiB` | `3.20x` smaller |
| V cache | `1792 MiB` | `560 MiB` | `3.20x` smaller |
| Median peak working set | `5662.69 MiB` | `3198.95 MiB` | `43.51%` lower |
| Median peak private memory | `3812.64 MiB` | `1337.93 MiB` | `64.91%` lower |

The allocation ratio matches the theoretical `16 / 5 = 3.2` ratio between F16 and the 5-bit Q4_HQQ block layout.

### 4.3 Output and speed check

A same-model, same-prompt smoke comparison produced coherent output for both cache formats.

| Metric | F16 cache | Q4_HQQ cache |
|---|---:|---:|
| Prompt processing | `57.8` tokens/s | `56.7` tokens/s |
| Generation | `2.54` tokens/s | `2.57` tokens/s |

These are single-run checks and are not used as a general performance claim.

---

## 5. Vulkan Backend

### 5.1 Implemented operations

The Vulkan backend includes Q4_HQQ support for:

- shader-visible block types and packed data access;
- F32/F16 to Q4_HQQ conversion;
- Q4_HQQ to F32 dequantization;
- matrix multiplication and model inference;
- cooperative-matrix dequantization;
- `MUL_MAT_ID` registration;
- Q4_HQQ K and V cache operations;
- Flash Attention with Q4_HQQ in K, V, or both;
- zero-initialized KV-cache blocks;
- type-size, stride, alignment, and operation support checks.

The dequantization formula is identical to the CPU implementation. The current cooperative-matrix helper computes the FP16 reciprocal of `scale` once and multiplies by it, avoiding repeated division in the active prompt-processing path.

### 5.2 Vulkan inference command

```powershell
.\build-vulkan\bin\Release\llama-cli.exe `
  -m .\models\llama-3.2-3b-instruct-q4_hqq.gguf `
  -p "What is bitcoin?" `
  -n 100 -t 10 `
  --device Vulkan0 -ngl 99 --fit off `
  --seed 42 --single-turn
```

The NVIDIA validation offloaded all model layers. Q4_HQQ model inference produced coherent output.

### 5.3 Vulkan cache command

```powershell
.\build-vulkan\bin\Release\llama-cli.exe `
  -m .\models\llama-3.2-3b-instruct-q4_hqq.gguf `
  -p "What is bitcoin?" `
  -n 100 -t 10 `
  --device Vulkan0 -ngl 99 `
  --flash-attn on `
  --cache-type-k q4_hqq `
  --cache-type-v q4_hqq `
  --seed 42 --single-turn
```

### 5.4 Backend correctness tests

The focused Vulkan suite was run on both NVIDIA and Intel adapters.

| Operation | Coverage per device | Result |
|---|---:|---|
| `MUL_MAT` | 3 shapes, including `3072 x 39 x 3072` | PASS on both devices |
| `CPY` | 4 small and large conversion cases | PASS on both devices |
| `FLASH_ATTN_EXT` | Q4_HQQ in K, V, and both | PASS on both devices |

The final all-operation Q4_HQQ suite passed 13/13 checks on each Vulkan device. The Q4_0 reference suite passed 29/29 checks.

---

## 6. Q4_HQQ vs Q4_0

### 6.1 Model size

| Format | File size | Quantizer data report | Relative size |
|---|---:|---:|---:|
| Q4_0 | `1,917,190,592` bytes | `1820.90 MiB`, `4.75 BPW` | baseline |
| Q4_HQQ | `2,093,351,360` bytes | `1988.90 MiB`, `5.19 BPW` | `+9.19%` |

The whole-model BPW values include tensors that remain in other formats according to the model quantization policy.

### 6.2 Final Windows throughput comparison

All headline values below are medians of seven raw samples with identical model files, `pp512`, `tg128`, ten CPU threads, and the same backend settings.

| Backend | Test | Q4_0 | Q4_HQQ | Q4_HQQ / Q4_0 |
|---|---|---:|---:|---:|
| CPU | `pp512` | `78.9888` | `44.7974` | `56.71%` |
| CPU | `tg128` | `9.81128` | `8.95402` | `91.26%` |
| NVIDIA Vulkan0 | `pp512` | `2737.24` | `910.791` | `33.27%` |
| NVIDIA Vulkan0 | `tg128` | `78.7885` | `66.8413` | `84.84%` |

Q4_HQQ generation speed is close to Q4_0, while prompt processing remains slower. On CPU, Q4_0 reaches an architecture-specific repacked SGEMM path that is not available for Q4_HQQ. On Vulkan, Q4_HQQ currently uses a correctness-oriented FP32 accumulation path.

### 6.3 Additional platform results

| Platform | Backend | `pp512` | `tg128` |
|---|---|---:|---:|
| Ubuntu WSL2, i7-12650H | Q4_HQQ CPU | `48.5529` | `8.83065` |
| Apple M1 | Q4_HQQ CPU/Accelerate | `103.851` | `4.39498` |
| Apple M1 | Q4_HQQ Vulkan/MoltenVK | `176.222` | `16.5233` |

The WSL, native Windows, and macOS results are reported separately because operating-system scheduling, compiler, storage path, and backend implementation differ.

### 6.4 Output quality

Both Q4_0 and Q4_HQQ produced coherent responses to:

- `What is bitcoin?`
- `Write a Python function to reverse a list.`

No visible quality regression was observed in these prompt checks.

### 6.5 Perplexity bonus

A bounded Wikitext-2 raw test used the first four 512-token chunks, CPU-only execution, ten threads, batch 512, ubatch 512, seed 42, and identical settings.

| Format | Perplexity | Wall time |
|---|---:|---:|
| Q4_0 | `9.4473 +/- 0.82340` | `33.099 s` |
| Q4_HQQ | `9.3079 +/- 0.80037` | `107.346 s` |

The uncertainty intervals overlap. The test did not detect a Q4_HQQ quality regression, but the sample is too small to claim an improvement.

---

## 7. Task 3: `--mmproj-backend`

### 7.1 Behavior

The option selects the device used by the multimodal projector:

```text
--mmproj-backend DEVICE
```

The base language model continues to use the devices selected by `--device`, `--n-gpu-layers`, and the normal model parameters.

The data flow is:

```text
common argument parser
  -> common_params.mmproj_backend
  -> mtmd_context_params.device
  -> clip_context_params.device
  -> projector backend initialization and scheduler
```

The projector device is not added to the base-model device list.

If the option is omitted, the existing default behavior is preserved. Invalid device names are rejected during parsing or initialization, and an explicitly requested device does not silently fall back to another accelerator.

### 7.2 Main source changes

- `common/arg.cpp`
- `common/common.h`
- `tools/mtmd/mtmd.h`
- `tools/mtmd/mtmd.cpp`
- `tools/mtmd/clip.h`
- `tools/mtmd/clip.cpp`
- `tools/mtmd/mtmd-cli.cpp`
- `tools/mtmd/debug/mtmd-debug.cpp`
- `tools/server/server-context.cpp`
- `tests/test-arg-parser.cpp`
- `tests/test-mtmd-c-api.c`

### 7.3 Device discovery

```powershell
.\build-vulkan\bin\Release\llama-mtmd-cli.exe --list-devices
.\build-vulkan\bin\Release\llama-mtmd-cli.exe --help
```

The help output contains `--mmproj-backend DEVICE`. Device names match those returned by backend enumeration, for example `CPU`, `Vulkan0`, `Vulkan1`, and `MTL0` where available.

### 7.4 Run examples

### Base model on NVIDIA Vulkan, projector on CPU

```powershell
.\build-vulkan\bin\Release\llama-mtmd-cli.exe `
  -m .\models\SmolVLM-256M-Instruct-Q8_0.gguf `
  --mmproj .\models\mmproj-SmolVLM-256M-Instruct-Q8_0.gguf `
  --image .\vision-test-731.png `
  -p "Describe the image." `
  --device Vulkan0 -ngl 99 `
  --mmproj-backend CPU `
  -n 128 --temp 0 --seed 42
```

### Base model on CPU, projector on NVIDIA Vulkan

```powershell
.\build-vulkan\bin\Release\llama-mtmd-cli.exe `
  -m .\models\SmolVLM-256M-Instruct-Q8_0.gguf `
  --mmproj .\models\mmproj-SmolVLM-256M-Instruct-Q8_0.gguf `
  --image .\vision-test-731.png `
  -p "Describe the image." `
  --device none -ngl 0 `
  --mmproj-backend Vulkan0 `
  -n 128 --temp 0 --seed 42
```

### Apple Silicon: base model on Metal, projector on CPU

```bash
./build-mac-metal/bin/llama-mtmd-cli \
  -m model.gguf \
  --mmproj mmproj.gguf \
  --image image.png \
  --device MTL0 -ngl 99 \
  --mmproj-backend CPU
```

### 7.5 End-to-end validation matrix

The Windows test used the official SmolVLM 256M GGUF model, matching projector, and a test image containing text and three colored shapes.

| Base model | Projector | Result |
|---|---|---|
| NVIDIA `Vulkan0` | CPU | PASS |
| CPU | NVIDIA `Vulkan0` | PASS |
| NVIDIA `Vulkan0` | NVIDIA `Vulkan0` | PASS |
| NVIDIA `Vulkan0` | Intel `Vulkan1` | PASS |
| NVIDIA `Vulkan0` | omitted/default | PASS |
| any | invalid device name | Clear error, exit 1 |
| any | missing option value | Parser error, exit 1 |

All successful runs encoded the image and produced a non-empty response identifying the red square, blue circle, and green triangle.

The macOS validation added these combinations:

| Base model | Projector | Result |
|---|---|---|
| Metal `MTL0` | CPU | PASS |
| CPU | Metal `MTL0` | PASS |
| MoltenVK `Vulkan0` | CPU | PASS |

This confirms that the projector backend can be selected independently of the base model backend.

---

## 8. Tests

### 8.1 Quantization tests

```powershell
.\build-cpu\bin\Release\test-quantize-q4-hqq.exe
.\build-cpu\bin\Release\test-quantize-fns.exe
```

Coverage includes:

- 20-byte block layout and 5.0-bit storage;
- nibble packing order;
- random, positive, negative, mixed, and multi-block data;
- all-zero and constant blocks;
- very small finite ranges;
- large finite offsets;
- FP16 metadata validity;
- fractional zero points;
- scalar and AVX2 dot products;
- one-row and two-row dispatch;
- raw zero KV-cache sentinel;
- rejection of invalid persisted blocks.

### 8.2 CPU backend tests

```powershell
.\build-cpu\bin\Release\test-backend-ops.exe `
  test -o MUL_MAT -b CPU -p q4_hqq
```

Focused CPU coverage passed for matrix multiplication, copy operations, and Flash Attention cache combinations.

### 8.3 Vulkan backend tests

```powershell
.\build-vulkan\bin\Release\test-backend-ops.exe `
  test -o MUL_MAT -b Vulkan0 -p q4_hqq

.\build-vulkan\bin\Release\test-backend-ops.exe `
  test -o CPY -b Vulkan0 -p q4_hqq

.\build-vulkan\bin\Release\test-backend-ops.exe `
  test -o FLASH_ATTN_EXT -b Vulkan0 -p q4_hqq
```

The same focused tests were run on the Intel Vulkan device.

### 8.4 Multimodal tests

```powershell
.\build-vulkan\bin\Release\test-arg-parser.exe
.\build-vulkan\bin\Release\test-mtmd-c-api.exe
```

Both tests pass in CPU and Vulkan builds. Default C API behavior remains unchanged when no projector device is supplied.

---

## 9. Deliverables

| Requirement | Included material |
|---|---|
| Task 1 conversion and Q4_0 quantization | Exact commands, conversion/quantization output, Q4_0 model metrics |
| Task 1 prompt outputs | Bitcoin and Python-list prompt logs |
| Task 1 latency, speed, and RAM | Benchmark and five-run memory results |
| Q4_HQQ implementation | Git history and patch/diff |
| Quantized Q4_HQQ model | Reproduction command; hashes and size recorded in manifests |
| Q4_HQQ vs Q4_0 comparison | Size, CPU/Vulkan speed, prompt quality, and perplexity tables |
| Q4_HQQ KV cache | CLI support, memory comparison, output checks, CPU/Vulkan tests |
| GPU implementation | Vulkan shaders, dispatch, model inference, cache inference, backend tests |
| SIMD bonus | AVX2 one-row and two-row CPU paths |
| Perplexity bonus | Limited Wikitext-2 comparison |
| `--mmproj-backend` | Source changes, commands, parser/C API tests, end-to-end logs |
| Setup instructions | Windows, WSL2, macOS CPU, Metal, and MoltenVK commands |

Raw build, test, benchmark, memory, inference, and multimodal logs are stored in the supplied validation archives. The Task 2 patch is also preserved as a standalone patch artifact.
