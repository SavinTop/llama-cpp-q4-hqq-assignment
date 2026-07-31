# Task 3 — `--mmproj-backend`

The option selects the multimodal projector backend independently from the base-model backend:

```text
--mmproj-backend DEVICE
```

## Example: base model on Vulkan, projector on CPU

```powershell
.\build-task2-vulkan\bin\Release\llama-mtmd-cli.exe `
  -m .\models\smolvlm-256m\SmolVLM-256M-Instruct-Q8_0.gguf `
  --mmproj .\models\smolvlm-256m\mmproj-SmolVLM-256M-Instruct-Q8_0.gguf `
  --image .\models\vision-test-731.png `
  --device Vulkan0 -ngl 99 `
  --mmproj-backend CPU `
  -p "Describe the image." -n 128
```

## Example: base model on CPU, projector on Vulkan

```powershell
.\build-task2-vulkan\bin\Release\llama-mtmd-cli.exe `
  -m .\models\smolvlm-256m\SmolVLM-256M-Instruct-Q8_0.gguf `
  --mmproj .\models\smolvlm-256m\mmproj-SmolVLM-256M-Instruct-Q8_0.gguf `
  --image .\models\vision-test-731.png `
  --device CPU -ngl 0 `
  --mmproj-backend Vulkan0 `
  -p "Describe the image." -n 128
```

## Validated behavior

| Base model | Projector | Result |
|---|---|---|
| Vulkan0 | CPU | PASS |
| CPU | Vulkan0 | PASS |
| Vulkan0 | Vulkan0 | PASS |
| Vulkan0, NVIDIA | Vulkan1, Intel | PASS |
| any | invalid projector device | clear error and non-zero exit |
| any | missing option value | parser error and non-zero exit |

Representative successful and error logs are stored under `logs/`.
