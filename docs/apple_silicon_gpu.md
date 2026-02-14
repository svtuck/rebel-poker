# Custom GPU Kernels on Apple Silicon

Triton does not support Apple Silicon (Metal backend only). Here are the
viable options for GPU-accelerated poker solving on Mac Studio.

## Framework Comparison

| Framework | Custom Kernels | Autodiff | Python API | Effort |
|-----------|---------------|----------|------------|--------|
| MLX       | Yes (Metal)   | Yes      | NumPy-like | Low    |
| PyTorch MPS | No          | Yes      | PyTorch    | Low    |
| Metal (PyObjC) | Yes      | No       | Verbose    | High   |

## 1. MLX (Recommended)

Apple's ML framework. Best balance of custom kernel support and usability.

```python
import mlx.core as mx
from mlx.core.fast import metal_kernel

# Example: vectorized regret matching across all infosets
source = """
    uint idx = thread_position_in_grid.x;
    float r = regrets[idx];
    float pos = r > 0.0f ? r : 0.0f;
    strategy[idx] = pos;  // unnormalized; normalize in Python
"""

regret_match = metal_kernel(
    name="regret_match",
    input_names=["regrets"],
    output_names=["strategy"],
    source=source,
)

# Usage: single GPU dispatch for all infosets
regrets = mx.array([...])  # all infoset regrets flattened
strategy = regret_match(
    inputs=[regrets],
    output_shapes=[regrets.shape],
    output_dtypes=[mx.float32],
    grid=(regrets.size, 1, 1),
    threadgroup=(256, 1, 1),
)
```

Key properties:
- **Unified memory**: no CPU<->GPU transfer cost
- **Lazy evaluation**: operations are batched and fused automatically
- `mlx.core.fast.metal_kernel()` for custom Metal Shading Language kernels
- Autodiff works through custom kernels via `mx.custom_function`

### CFR-specific MLX patterns

For poker CFR, the key GPU-friendly operations are:

1. **Regret matching**: embarrassingly parallel across infosets
2. **Reach probability propagation**: matrix multiply (deals x actions)
3. **Belief state normalization**: reduce + broadcast
4. **Value network forward/backward**: standard NN ops

```python
# Vectorized belief state computation in MLX
def compute_pbs(chance_probs, reach_p0, reach_p1):
    """All operations are lazy — MLX fuses them into one GPU dispatch."""
    joint = chance_probs * reach_p0 * reach_p1
    return joint / joint.sum()
```

## 2. PyTorch MPS Backend

Best for value/policy network training where you want standard PyTorch APIs.

```python
import torch
device = torch.device("mps")

# Value network training on Apple GPU
model = ValueNetwork().to(device)
pbs = torch.randn(batch_size, num_deals, device=device)
values = model(pbs)
```

Limitations:
- No custom kernel API (as of PyTorch 2.x)
- Some ops not yet supported on MPS (check `torch.backends.mps.is_available()`)
- Good for NN training, not ideal for CFR tree traversal

## 3. Metal Compute Shaders (via PyObjC)

Full control but high complexity. Use only for performance-critical paths
that MLX can't handle.

```python
import Metal

device = Metal.MTLCreateSystemDefaultDevice()
source = """
#include <metal_stdlib>
using namespace metal;

kernel void cfr_update(
    device float* regrets [[buffer(0)]],
    device float* strategy [[buffer(1)]],
    device float* reach    [[buffer(2)]],
    constant uint& n_infosets [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n_infosets) return;

    // Regret matching for 2-action infosets
    uint base = id * 2;
    float r0 = max(regrets[base], 0.0f);
    float r1 = max(regrets[base + 1], 0.0f);
    float total = r0 + r1;

    if (total > 0.0f) {
        strategy[base] = r0 / total;
        strategy[base + 1] = r1 / total;
    } else {
        strategy[base] = 0.5f;
        strategy[base + 1] = 0.5f;
    }
}
"""

result = device.newLibraryWithSource_options_error_(source, None, None)
library = result[0]
fn = library.newFunctionWithName_("cfr_update")
pipeline = device.newComputePipelineStateWithFunction_error_(fn, None)[0]
```

## Recommended Architecture for ReBeL on Mac Studio

```
+------------------------------------------+
|  Python Orchestration Layer               |
|  (game tree structure, training loop)     |
+------------------------------------------+
        |                      |
+----------------+   +------------------+
| MLX Backend    |   | PyTorch MPS      |
| - CFR updates  |   | - Value network  |
| - Reach probs  |   | - Policy network |
| - PBS compute  |   | - Training loop  |
| - Custom Metal |   |                  |
+----------------+   +------------------+
        |                      |
+------------------------------------------+
|  Apple Silicon Unified Memory             |
|  (M2 Ultra: 76 GPU cores, 192GB)         |
+------------------------------------------+
```

### Migration path from current code

1. **Phase 1 (now)**: PyTorch CPU — validates correctness
2. **Phase 2**: PyTorch MPS — GPU-accelerate value net training
3. **Phase 3**: MLX — port CFR tensor ops for GPU-native solving
4. **Phase 4**: Custom Metal kernels via MLX for hot paths

### Performance expectations

- Kuhn Poker: too small to benefit from GPU (6 deals, ~12 infosets)
- Leduc Poker: moderate benefit (936 deals, ~936 infosets)
- NLHE River: significant benefit (millions of states, GPU parallelism essential)
- The main win is parallelizing across deals/hands in the vectorized CFR
