# Short Assignment Report

## Work completed

The assignment was implemented against `llama.cpp` revision `555881ebc8b0fc0402b30e09258a32a7bfd13c52` and validated on Windows 11, Ubuntu WSL2, and macOS on Apple M1.

For Task 1, Llama 3.2 3B Instruct was downloaded from Hugging Face, converted to F16 GGUF with the repository conversion script, and quantized to Q4_0 with `llama-quantize`. The model was tested with the requested Bitcoin and Python-list prompts. Both responses were coherent. On the Windows CPU test system, prompt processing reached `77.71 +/- 4.26` tokens/s, token generation reached `8.92 +/- 0.26` tokens/s, and single-token evaluation latency was `120.5 ms`. The five-run average peak working set was approximately `3534 MiB`.

For Task 2, a new asymmetric 4-bit quantization type, `Q4_HQQ`, was added. The implementation includes the 20-byte block structure, quantization and dequantization, Q4_HQQ x Q8_0 dot products, AVX2 execution, GGUF metadata, model loading, quantizer integration, and CLI support. Q4_HQQ was also added as a KV-cache type through `--cache-type-k q4_hqq` and `--cache-type-v q4_hqq`. Vulkan support covers model inference, tensor conversion, matrix operations, and Q4_HQQ K/V cache use with Flash Attention.

The Q4_HQQ model was `9.19%` larger than Q4_0. In the final seven-sample Windows comparison, Q4_HQQ reached `56.71%` of Q4_0 CPU prompt-processing throughput and `91.26%` of its generation throughput. On NVIDIA Vulkan, it reached `33.27%` of Q4_0 prompt-processing throughput and `84.84%` of generation throughput. Both formats produced coherent output. A limited Wikitext-2 test produced perplexity values of `9.4473 +/- 0.82340` for Q4_0 and `9.3079 +/- 0.80037` for Q4_HQQ; the intervals overlap, so no quality improvement is claimed.

At a 32768-token context, Q4_HQQ K/V cache reduced the allocated cache from `3584 MiB` to `1120 MiB`, matching the expected `3.2x` reduction from F16 to the 5-bit block layout. Output remained coherent with both cache formats.

For Task 3, `--mmproj-backend DEVICE` was added to the multimodal inference path. It selects the backend used by the multimodal projector without changing the base-model backend. The option was tested with CPU, NVIDIA Vulkan, Intel Vulkan, native Metal, and MoltenVK combinations. Invalid device names fail with a clear error instead of silently selecting another backend.

## Problems solved during implementation

The main correctness problems were FP16 overflow in stored quantization metadata, preservation of the fractional zero point, handling zero-initialized KV-cache blocks, missing Vulkan pipeline registrations, and an incorrect experimental Vulkan integer-dot path. The integer-dot optimization was removed after it failed a large backend comparison. The final implementation keeps the validated reciprocal-based Vulkan dequantization path.

## Possible next steps

The largest remaining performance gap is prompt processing. Q4_HQQ does not yet use the architecture-specific repacked CPU matrix path available to Q4_0, and the Vulkan path still prioritizes correctness over maximum throughput. Reasonable next steps are a validated repacked CPU kernel, an ARM64 NEON dot-product implementation, broader perplexity testing, and longer multi-sequence KV-cache tests.
