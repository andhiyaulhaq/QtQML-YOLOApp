Phase 3 Sketch: Memory Optimization

Purpose
- Reduce dynamic allocations and maximize reuse of memory buffers to improve throughput and reduce fragmentation.

Assumptions
- A pool of reusable buffers for frames, tensors, and intermediate results is feasible within the system memory budget.
- Inference remains tied to the Phase 1 session pool for multi-session inference.

Deliverables
- FrameBufferPool: preallocated frame buffers with RAII ownership.
- TensorBufferPool: reuse inference tensors across inferences and sessions.
- Move semantics everywhere: reduce copies, leverage std::move where possible.
- Optional: OpenCV UMat or other zero-copy approaches for GPU paths.
- Memory usage profiling and logging hooks to validate reuse effectiveness.

Implementation Plan
- Phase 3A: FramePool
  - Preallocate a fixed number of cv::Mat buffers for incoming frames and intermediate results.
  - Provide acquire/release semantics to producers/consumers.
- Phase 3B: TensorPool
  - Preallocate tensor buffers per session or per inference context.
  - Ensure tensors can be reused safely across inferences (with per-item lifetimes).
- Phase 3C: Move Semantics & Zero-Copy
  - Replace copies with moves; pass buffers and tensors by rvalue when possible.
  - Examine hot paths for potential cv::Mat::clone() avoidance and use of data ownership transfers.
- Phase 3D: Optional GPU Backends
  - If GPU path is used, explore cv::UMat and/or CUDA streams to reduce host-device copies.
- Phase 3E: Profiling & Validation
  - Track allocations, deallocations, and reuse counts; measure impact on latency and throughput.

Risks & Mitigations
- Memory budget risk: tune pool sizes to match available RAM/VRAM.
- Complexity: keep these abstractions small and well-documented to avoid maintenance burden.
- Compatibility: ensure existing interfaces can still work with pooled buffers.

Integration Points
- Inference: src/inference.h/.cpp (TensorPool, FramePool)
- VideoController: src/VideoController.* (pipeline and memory management references)
- Main / App lifecycle: ensure memory pools are initialized and destroyed with app.

Diagram (ASCII)
```
┌ Phase 3: Memory Optimization ───────────────────────────────────────┐
│                                                                  │
│  FramePool        TensorPool     TensorMove/Reuse             │
│  ┌----------┐   ┌-----------┐   ┌-----------┐                │
│  │ Buffers  │   │ Buffers   │   │ Moved/Owned│                │
│  └----------┘   └-----------┘   └-----------┘                │
│        │               │               │                    │
│        ▼               ▼               ▼                    │
│  Frame/Blob transfer between stages with minimal copies        │
│                                                                  │
│  Inference uses Phase 1 Session Pool with reusable tensors        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

End of Sketch
