# llama.cpp Q4_HQQ Assignment

This repository is based on `llama.cpp` commit:

```text
555881ebc8b0fc0402b30e09258a32a7bfd13c52
```

The implementation adds:

- a new affine 4-bit quantization format, `Q4_HQQ`;
- scalar and AVX2 CPU execution paths;
- multi-row CPU prompt-processing optimization;
- GGUF conversion and model quantization support;
- Q4_HQQ KV-cache support;
- Vulkan inference support;
- multimodal projector backend selection through `tools/mtmd`;
- correctness tests and reproducible benchmark commands.

## Q4_HQQ format

Q4_HQQ stores 32 quantized weights in a 20-byte block:

```c
#define QK4_HQQ 32

typedef struct {
    ggml_half scale;
    ggml_half zero;
    uint8_t qs[QK4_HQQ / 2];
} block_q4_hqq;
```

The resulting storage cost is:

```text
20 bytes / 32 weights = 5 bits per weight
```

### Quantization

For each block:

```text
scale = 15 / (max - min)
zero  = -min * scale
q     = clamp(round(w * scale + zero), 0, 15)
```

### Dequantization

```text
w = (q - zero) / scale
```

### Q4_HQQ × Q8_0 dot product

```text
(d8 / scale4) *
(sum(q4_code * q8_code) - zero4 * sum(q8_code))
```

The Q4_HQQ zero point remains floating point throughout the computation.

The implementation handles:

- all-zero blocks;
- constant positive and negative blocks;
- very small finite ranges;
- large finite offsets;
- FP16 metadata limits;
- exact all-zero KV-cache blocks;
- low/high nibble packing for 32-element blocks.

## Implementation overview

### CPU backend

The CPU implementation includes:

- portable scalar quantization and dequantization;
- portable scalar Q4_HQQ × Q8_0 dot product;
- AVX2 dot-product acceleration on supported x86 processors;
- multi-row AVX2 execution used during prompt processing;
- scalar fallback for unsupported processors and row layouts.

### GGUF and model quantization

Q4_HQQ is integrated into:

- GGML type registration;
- GGUF tensor serialization;
- model quantization;
- tensor size and block-size calculations;
- CPU backend operation dispatch;
- model loading and inference.

Models can be quantized directly to `Q4_HQQ` with `llama-quantize`.

### KV cache

Q4_HQQ can be selected for KV-cache tensors through the standard cache type options.

Supported configurations include:

- Q4_HQQ K and V with Flash Attention enabled;
- Q4_HQQ K with F16 V without Flash Attention;
- CPU and Vulkan execution paths.

The implementation also defines handling for backend-cleared KV memory using an exact all-zero block representation.

### Vulkan backend

The Vulkan implementation includes:

- Q4_HQQ type registration;
- quantization shaders;
- dequantization shaders;
- matrix multiplication support;
- copy and conversion operations;
- Flash Attention support;
- KV-cache execution support;
- shader generation and backend dispatch.

### Multimodal projector backend

`tools/mtmd` supports an independent projector backend option:

```text
--mmproj-backend <device>
```

This option selects the device used by the multimodal projector without changing the backend selected for the base language model.

Example with the base model on Vulkan and the projector on CPU:

```bash
llama-mtmd-cli \
  -m model.gguf \
  --mmproj mmproj.gguf \
  --image image.png \
  --device Vulkan0 \
  --n-gpu-layers all \
  --mmproj-backend CPU
```

Example with the base model on CPU and the projector on Vulkan:

```bash
llama-mtmd-cli \
  -m model.gguf \
  --mmproj mmproj.gguf \
  --image image.png \
  --device none \
  --n-gpu-layers 0 \
  --mmproj-backend Vulkan0
```

An explicitly requested projector device is validated during initialization. Invalid device names are rejected instead of silently selecting another backend.

---

# Build

## Windows CPU

Run from a Visual Studio Developer PowerShell:

```powershell
cmake -S . -B build-cpu `
  -DGGML_NATIVE=ON `
  -DGGML_VULKAN=OFF

cmake --build build-cpu `
  --config Release `
  --parallel
```

Release binaries are placed under:

```text
build-cpu/bin/Release/
```

## Windows Vulkan

Install the Vulkan SDK and configure a Vulkan build:

```powershell
cmake -S . -B build-vulkan `
  -DGGML_NATIVE=ON `
  -DGGML_VULKAN=ON

cmake --build build-vulkan `
  --config Release `
  --parallel
```

Release binaries are placed under:

```text
build-vulkan/bin/Release/
```

## Linux or WSL2 CPU

Install the required build tools:

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build
```

Configure and build:

```bash
cmake -S . -B build-linux -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DGGML_VULKAN=OFF

cmake --build build-linux --parallel
```

## macOS Apple Silicon CPU

Install the command-line tools and build dependencies:

```bash
xcode-select --install
brew install cmake ninja
```

Configure and build:

```bash
cmake -S . -B build-mac-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DGGML_METAL=OFF \
  -DGGML_VULKAN=OFF

cmake --build build-mac-cpu --parallel
```

## macOS Apple Silicon Metal

Configure a native Metal build for multimodal backend testing:

```bash
cmake -S . -B build-mac-metal -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DGGML_METAL=ON \
  -DGGML_VULKAN=OFF

cmake --build build-mac-metal --parallel \
  --target llama-mtmd-cli test-arg-parser test-mtmd-c-api
```

## macOS Apple Silicon Vulkan through MoltenVK

Install the Vulkan SDK for macOS with MoltenVK, then activate its environment in the current terminal.

For a standard LunarG Vulkan SDK installation:

```bash
source "$HOME/VulkanSDK/<version>/setup-env.sh"
```

Confirm that Vulkan tools can see the MoltenVK device:

```bash
vulkaninfo --summary
```

Configure a Vulkan-only build:

```bash
cmake -S . -B build-mac-vulkan -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DGGML_METAL=OFF \
  -DGGML_VULKAN=ON

cmake --build build-mac-vulkan --parallel \
  --target llama-cli llama-bench llama-mtmd-cli \
           test-backend-ops test-arg-parser test-mtmd-c-api
```

List devices detected by the Vulkan build:

```bash
./build-mac-vulkan/bin/llama-bench --list-devices
```

The MoltenVK device is normally exposed as `Vulkan0`.

---

# Model conversion and quantization

## Convert a Hugging Face model to GGUF

```bash
python convert_hf_to_gguf.py \
  /path/to/hugging-face-model \
  --outfile /path/to/model-f16.gguf \
  --outtype f16
```

## Quantize to Q4_HQQ

```bash
llama-quantize \
  /path/to/model-f16.gguf \
  /path/to/model-q4_hqq.gguf \
  Q4_HQQ
```

On Windows:

```powershell
.\build-cpu\bin\Release\llama-quantize.exe `
  C:\path\to\model-f16.gguf `
  C:\path\to\model-q4_hqq.gguf `
  Q4_HQQ
```

---

# Correctness tests

The examples below use Unix-style binary paths. For a Windows multi-configuration build, use the corresponding executable under `bin/Release`.

## Q4_HQQ quantization tests

```bash
./build/bin/test-quantize-q4-hqq
```

This test covers Q4_HQQ-specific cases including:

- quantization and dequantization;
- constant blocks;
- zero blocks;
- fractional zero points;
- very small finite ranges;
- large finite offsets;
- metadata encoding;
- scalar and optimized dot-product behavior.

## General quantization tests

```bash
./build/bin/test-quantize-fns
```

## CPU matrix multiplication

```bash
./build/bin/test-backend-ops \
  test \
  -o MUL_MAT \
  -b CPU \
  -p q4_hqq
```

## Vulkan matrix multiplication

```bash
./build/bin/test-backend-ops \
  test \
  -o MUL_MAT \
  -b Vulkan0 \
  -p q4_hqq
```

## Multimodal argument parser

```bash
./build/bin/test-arg-parser
```

## Multimodal C API

```bash
./build/bin/test-mtmd-c-api
```

## Windows CPU tests

```powershell
.\build-cpu\bin\Release\test-quantize-q4-hqq.exe

.\build-cpu\bin\Release\test-quantize-fns.exe

.\build-cpu\bin\Release\test-backend-ops.exe `
  test `
  -o MUL_MAT `
  -b CPU `
  -p q4_hqq
```

## Windows Vulkan tests

```powershell
.\build-vulkan\bin\Release\test-backend-ops.exe `
  test `
  -o MUL_MAT `
  -b Vulkan0 `
  -p q4_hqq

.\build-vulkan\bin\Release\test-arg-parser.exe
.\build-vulkan\bin\Release\test-mtmd-c-api.exe
```

## macOS CPU tests

```bash
./build-mac-cpu/bin/test-quantize-q4-hqq
./build-mac-cpu/bin/test-quantize-fns

./build-mac-cpu/bin/test-backend-ops \
  test \
  -o MUL_MAT \
  -b CPU \
  -p q4_hqq
```

## macOS Vulkan tests

```bash
./build-mac-vulkan/bin/test-backend-ops \
  test \
  -o MUL_MAT \
  -b Vulkan0 \
  -p q4_hqq

./build-mac-vulkan/bin/test-arg-parser
./build-mac-vulkan/bin/test-mtmd-c-api
```

---

# Inference

## CPU inference

```bash
llama-cli \
  -m /path/to/model-q4_hqq.gguf \
  -ngl 0 \
  -p "Explain virtual memory." \
  -n 128
```

Windows:

```powershell
.\build-cpu\bin\Release\llama-cli.exe `
  -m C:\path\to\model-q4_hqq.gguf `
  -ngl 0 `
  -p "Explain virtual memory." `
  -n 128
```

## Vulkan inference

```bash
llama-cli \
  -m /path/to/model-q4_hqq.gguf \
  -ngl 99 \
  -p "Explain virtual memory." \
  -n 128
```

Windows:

```powershell
.\build-vulkan\bin\Release\llama-cli.exe `
  -m C:\path\to\model-q4_hqq.gguf `
  -ngl 99 `
  -p "Explain virtual memory." `
  -n 128
```

macOS through MoltenVK:

```bash
./build-mac-vulkan/bin/llama-cli \
  -m /path/to/model-q4_hqq.gguf \
  -ngl 99 \
  -p "Explain virtual memory." \
  -n 128
```
