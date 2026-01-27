Development Plan: Phases 1–4 for YOLOApp Multi-Core Optimization

Overview
- Goal: exhaustively utilize all CPU cores for YOLO inference in a production-ready, maintainable way.
- Scope: phases cover threading, session pooling, multi-stage pipeline, memory optimization, and thread-safety.
- Assumptions: existing codebase (ONNX Runtime, OpenCV, Qt UI) remains the integration point; hardware is multi-core CPU with optional CUDA.

Phase 1 Recap (Merged Phase 0)
- The codebase already configures ONNX Runtime threading and graph optimizations.
- Phase 1 now includes a multi-session pool with round-robin dispatch to fully utilize all CPU cores.
- The single-session bottleneck is eliminated by pooling and load balancing to improve throughput.

Phase 1: ONNX Runtime Threading + Session Pool (Completion Target)
Objective
- Replace the single session with a pool of Ort::Session instances and distribute inferences across them to improve throughput and reduce tail latency.

Assumptions
- User can set: intraOpNumThreads, interOpNumThreads, sessionPoolSize (>=1).
- IO names (inputNodeNames, outputNodeNames) are identical across pool members.

Deliverables
- A working session pool (size = sessionPoolSize) with round-robin load balancing.
- Warm-up across all pool sessions to stabilize startup costs.
- Safe cleanup of all pooled sessions on shutdown.
- Minimal public API surface impact; existing calls to RunSession work with a pool transparently.

Implementation Plan
- Data model changes (Phase 1):
  - Add: std::vector<Ort::Session*> m_sessionPool; std::atomic<size_t> m_sessionIndex;
  - Reuse inputNodeNames/outputNodeNames across pool members.
- Initialization:
  - In CreateSession, create N sessions as per sessionPoolSize and push into m_sessionPool.
  - Set primary session indicator for backward compatibility if needed.
- Inference path:
  - At each inference, select session = m_sessionPool[m_index % poolSize] via atomic fetch_add.
- Warm-up:
  - Run a small warm-up pass on every pool member.
- Cleanup:
  - Destroy all Ort::Session instances in the pool on destructor.
- Testing & validation:
  - Baseline tests to compare pre/post Phase-1 throughput and latency.

Risks & Mitigations
- Higher memory usage: align pool size with RAM/VRAM; start small (2–4).
- Multi-thread safety: ensure input/output names are managed as read-only across pool.
- Startup cost: warm-up reduces tail latency; consider lazy initialization if needed.

Phase 2: Production Pipeline Architecture (Multi-Stage)
Objective
- Introduce a multi-stage threading model to decouple Capture, Preprocess, Inference, Postprocess, and Display.

Assumptions
- A thread-per-stage approach with lock-free queues between stages.
- Inference stage uses the Phase 1 pool to maximize throughput.

Deliverables
- A pipeline with at least 2 Capture threads, multiple Preprocess, 1–3 Inference threads (pool-based), 2 Postprocess threads, and 1 Display thread.
- Lock-free or low-contention queues between stages with backpressure.
- Per-stage metrics collection.

Implementation Plan
- Add a small, generic Queue<T> abstraction (SPSC/MPSC as needed).
- Refactor VideoController/CameraWorker to feed frames into a shared pipeline, with per-stage workers.
- Align data ownership with zero-copy strategies where possible (preallocated buffers, move semantics).
- Integrate mild backpressure to avoid overflow.
- Validation: measure end-to-end latency and max sustainable FPS.

Risks & Mitigations
- Increased code complexity; add clear abstractions and unit tests.
- Thread lifecycle management; ensure clean shutdown to avoid data races.

Phase 3: Memory Optimization
Objective
- Reduce dynamic allocations and leverage object pools to minimize GC pressure and fragmentation.

Assumptions
- Availability of reusable frame buffers and inference tensors.

Deliverables
- FramePool and TensorPool with reuse semantics.
- Optional use of cv::UMat for GPU-accelerated regions.
- Zero-copy data movement where feasible.

Implementation Plan
- Implement FramePool: preallocate a pool of cv::Mat buffers; recycle after use.
- Implement TensorPool per session or per inference context; reuse input tensors.
- Introduce move semantics in data paths; minimize cv::Mat.clone() calls.
- Optional: adopt cv::UMat for GPU backends where supported.
- Validation: track heap allocations and reuse effectiveness; measure allocation counts per frame.

Risks & Mitigations
- Memory spend if pools are too large; tune pool sizes.
- Compatibility with existing code paths; ensure no ABI/DI mismatches.

Phase 4: Thread Safety & Synchronization
Objective
- Harden the system against race conditions and race-related bugs; ensure robust, lock-free communication where feasible.

Assumptions
- The majority of hot paths should be lock-free; non-critical sections may use fine-grained atomics.

Deliverables
- Lock-free queues across all critical paths; thread-local storage for per-thread resources.
- Atomic counters for metrics; error handling and propagation through pipeline.
- Comprehensive tests for concurrency scenarios.

Implementation Plan
- Replace mutex-based paths with lock-free SPSC/MPSC queues where appropriate.
- Introduce per-session thread-local resources; avoid shared mutable state.
- Implement a centralized error propagation mechanism and health checks.
- Validation: concurrency stress tests; observe deadlocks, livelocks, data races using sanitizers.

Integration & Testing Strategy
- Build and unit tests for each phase component.
- End-to-end integration tests on a representative dataset.
- Performance tests measuring:
  - Throughput (FPS)
  - End-to-end latency
  - Tail latency (p95/p99)
  - CPU usage across all cores
- Regression tests to ensure existing features remain functional.

Build & Deployment Considerations
- Update CMake to expose new libraries and compile options if necessary.
- Document new configuration knobs (e.g., sessionPoolSize, per-stage thread counts).
- Provide a small runbook for developers to enable/disable phases.

Appendix: Cross-References
- Inference: src/inference.h/.cpp (Session pool, RunSession, WarmUpSession)
- VideoController: src/VideoController.h/.cpp (pipeline glue)
- Main: main.cpp (entry point and app lifecycle)
- Docs: this file at docs/DEVELOPMENT_PLAN_PHASES.md

End of Plan
