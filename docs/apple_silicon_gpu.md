# Custom GPU Kernels on Apple Silicon

Since Triton doesn't support Apple Silicon, here are the frameworks for writing
custom GPU kernels on Mac Studio:

## 1. MLX (Recommended for ML workloads)

Apple's ML framework with native Apple Silicon support.

```python
import mlx.core as mx
from mlx.core.fast import metal_kernel

# Custom Metal kernel via MLX
source = """
    uint index = thread_position_in_grid.x;
    if (index < input.size()) {
        output[index] = input[index] * scale;
    }
"""

kernel = metal_kernel(
    name="scale_kernel",
    input_names=["input"],
    output_names=["output"],
    source=source,
)
```

Key advantages:
- Unified memory (no CPU<->GPU copies)
- NumPy-like API
- `mlx.core.fast.metal_kernel()` for custom kernels
- Automatic differentiation support

## 2. Metal Compute Shaders (Full control)

Write kernels in Metal Shading Language (MSL), a C++14-based language.

```metal
// kernel.metal
kernel void cfr_update(
    device float* regrets [[buffer(0)]],
    device float* values [[buffer(1)]],
    device float* strategy [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    float positive_regret = max(regrets[id], 0.0f);
    // ... CFR regret matching logic
}
```

Access from Python via PyObjC:
```python
import Metal
device = Metal.MTLCreateSystemDefaultDevice()
library = device.newLibraryWithSource_options_error_(kernel_source, None, None)
```

## 3. PyTorch MPS Backend

```python
import torch
device = torch.device("mps")  # Apple GPU
tensor = torch.randn(1000, device=device)
```

Supports standard tensor operations. Limited custom kernel support but growing.

## Recommendation for ReBeL

1. **Start with PyTorch MPS** for standard tensor operations (belief state
   tracking, value network training, strategy computation)

2. **Use MLX metal_kernel()** for custom operations:
   - Vectorized CFR updates across all information sets
   - Batch reach probability propagation
   - Belief state normalization across large deal sets

3. **Consider pure Metal** only if MLX kernels aren't flexible enough
   for complex CFR tree traversals

## Performance Notes

- Mac Studio M2 Ultra: 76 GPU cores, 192GB unified memory
- Unified memory means no PCIe transfer overhead
- For poker tree traversal, the bottleneck is often tree structure (branching),
  not raw compute — so batching across deals/hands is key
- MLX lazy evaluation can help batch multiple operations
