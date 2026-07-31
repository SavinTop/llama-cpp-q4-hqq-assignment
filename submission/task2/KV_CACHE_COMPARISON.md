# Q4_HQQ K/V-Cache Comparison

The same Q4_HQQ weight model was run with a 32768-token context, Flash Attention enabled, ten CPU threads and three fresh processes per cache format.

| Metric | F16 K/V | Q4_HQQ K/V | Difference |
|---|---:|---:|---:|
| Explicit K/V allocation | 3584 MiB | 1120 MiB | -68.75%, 3.20x smaller |
| Average peak working set | 5662.94 MiB | 3198.93 MiB | -43.51% |

Both cache configurations produced coherent output. Q4_HQQ K/V execution completed successfully on CPU and Vulkan.

Raw process-memory and allocation evidence is stored in this directory.
