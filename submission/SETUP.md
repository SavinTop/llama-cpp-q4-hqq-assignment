# Setup Instructions

## Primary environment

| Item | Value |
|---|---|
| Operating system | Windows 11 Pro x64 |
| CPU | Intel Core i7-12650H |
| Logical processors | 16 |
| Benchmark threads | 10 |
| Installed RAM | 31.73 GiB |
| Compiler | MSVC 19.51 |
| Build system | CMake and Ninja or the configured MSVC generator |
| Vulkan SDK | 1.4.357 |
| Vulkan0 | NVIDIA GeForce RTX 3050 Ti Laptop GPU |
| Vulkan1 | Intel UHD Graphics |

## CPU build

```powershell
cmake -S . -B build-task2-cpu -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DGGML_NATIVE=ON `
  -DGGML_CUDA=OFF `
  -DGGML_VULKAN=OFF `
  -DLLAMA_BUILD_TESTS=ON `
  -DLLAMA_BUILD_TOOLS=ON

cmake --build build-task2-cpu --parallel
```

## Vulkan build

Run from a shell with the Vulkan SDK and MSVC environment available:

```powershell
cmake -S . -B build-task2-vulkan -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DGGML_NATIVE=ON `
  -DGGML_CUDA=OFF `
  -DGGML_VULKAN=ON `
  -DLLAMA_BUILD_TESTS=ON `
  -DLLAMA_BUILD_TOOLS=ON

cmake --build build-task2-vulkan --parallel
```

## Linux or WSL CPU build

```bash
cmake -S . -B build-linux -G Ninja   -DCMAKE_BUILD_TYPE=Release   -DGGML_NATIVE=ON   -DGGML_VULKAN=OFF   -DLLAMA_BUILD_TESTS=ON   -DLLAMA_BUILD_TOOLS=ON

cmake --build build-linux --parallel
```

## macOS MoltenVK build

LunarG SDK:

```bash
source "$HOME/VulkanSDK/<version>/setup-env.sh"
vulkaninfo --summary
```

Homebrew MoltenVK:

```bash
export VK_DRIVER_FILES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json
vulkaninfo --summary
```

Build:

```bash
cmake -S . -B build-mac-vulkan -G Ninja   -DCMAKE_BUILD_TYPE=Release   -DGGML_NATIVE=ON   -DGGML_METAL=OFF   -DGGML_VULKAN=ON   -DLLAMA_BUILD_TESTS=ON   -DLLAMA_BUILD_TOOLS=ON

cmake --build build-mac-vulkan --parallel
```

## Final focused validation

```powershell
.\submission\tools\Run-FinalValidation.ps1 -Requantize
```

This generates current logs under `submission/final/logs` and writes `submission/final/FINAL_VALIDATION.md`.
