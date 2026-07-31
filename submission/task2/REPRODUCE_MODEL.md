# Reproduce the Q4_HQQ Model

The quantized model can be reproduced from the Task 1 F16 GGUF. A model binary is intentionally not stored in Git.

```powershell
.\build-task2-cpu\bin\Release\llama-quantize.exe `
  .\models\llama-3.2-3b-instruct-f16.gguf `
  .\models\llama-3.2-3b-instruct-q4_hqq-final.gguf `
  Q4_HQQ
```

Expected final output:

```text
File: llama-3.2-3b-instruct-q4_hqq-final.gguf
Size: 2,016,397,760 bytes
SHA-256: FD44187FD1BA53C383F3D7B1BD1F0D5C2FB5EA48D23526170814DE563121903E
```

The ordinary command must quantize `token_embd.weight` to Q4_HQQ; no token-embedding override is required.

## CPU inference

```powershell
.\build-task2-cpu\bin\Release\llama-cli.exe `
  -m .\models\llama-3.2-3b-instruct-q4_hqq-final.gguf `
  -ngl 0 --single-turn `
  -p "What is bitcoin?" -n 100
```

## Vulkan inference

```powershell
.\build-task2-vulkan\bin\Release\llama-cli.exe `
  -m .\models\llama-3.2-3b-instruct-q4_hqq-final.gguf `
  -ngl 99 --single-turn `
  -p "What is bitcoin?" -n 100
```

## Q4_HQQ K/V cache

```powershell
.\build-task2-vulkan\bin\Release\llama-cli.exe `
  -m .\models\llama-3.2-3b-instruct-q4_hqq-final.gguf `
  -ngl 99 --single-turn `
  --flash-attn on `
  --cache-type-k q4_hqq `
  --cache-type-v q4_hqq `
  -p "What is bitcoin?" -n 100
```
