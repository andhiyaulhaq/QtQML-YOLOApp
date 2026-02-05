Phase 2 Sketch: Multi-Stage Production Pipeline

Purpose

- Sketch a production-ready, multi-stage pipeline to fully exploit multi-core CPUs for YOLO inference.
- Separate concerns across Capture, Preprocess, Inference, Postprocess, and Display with concurrent workers and lock-free queues.

High-Level Architecture

- Stages and rough thread allocations (configurable):
  - Capture: 2 threads (camera I/O)
  - Preprocess: 4 threads (CPU cores/2)
  - Inference: 2–3 threads (CPU cores/3) using the ONNX Runtime session pool
  - Postprocess: 2 threads (CPU cores/4)
  - Display: 1 thread (UI updates)

- Data flow uses a chain of lock-free queues between adjacent stages:
  - FrameQueue: Capture -> Preprocess
  - PreprocQueue: Preprocess -> Inference
  - InferenceQueue: Inference -> Postprocess
  - PostprocQueue: Postprocess -> Display
  - All queues are bounded (to apply backpressure) and lock-free to minimize contention.

- Inference uses the existing Phase 1 multi-session pool (sessionPoolSize) and a round-robin or per-session load balancing policy.
- Memory and data ownership are carefully managed to enable zero-copy transfers where possible (preallocated buffers, move semantics).

Key Components (reference points)

- YOLO_V8 (src/inference.h/.cpp): session pool, RunSession, WarmUpSession, tensor processing
- VideoController (src/VideoController.h/.cpp): pipeline orchestration and thread management
- CameraWorker (inside VideoController.cpp): frame capture and initial framing
- SystemMonitor: optional integration for runtime health checks

Threading & Synchronization

- Prefer lock-free queues between adjacent stages; use atomics for counters and state signals.
- Implement backpressure: if a stage's queue is full, upstream producers slow down or skip frames to avoid memory blow-up.
- Graceful shutdown: stop signals propagate through the pipeline; ensure in-flight data is completed or safely discarded.

Data Model & Memory

- Reuse frame buffers across stages; avoid unnecessary copies; move data where possible.
- Tensor and blob buffers reused from a pool; avoid allocations in hot paths.

Configuration Knobs (examples)

- phase2.enabled: true/false
- stage-thread-counts: per-stage thread counts (e.g., capture=2, preprocess=4, infer=3, postprocess=2, display=1)
- queue-sizes: per-queue capacity (e.g., 8–64 frames)
- sessionPoolSize: maintained from Phase 1 (e.g., 2–4)
- backpressure.enabled: true/false

Validation & Metrics

- Target metrics: end-to-end latency, per-stage latency, FPS, dropped frames, CPU utilization per core.
- Add lightweight tracing: queue depth, throughput per stage, and session pool utilization.
- Stress test with bursty inputs to validate backpressure behavior and stability.

Implementation Plan (high-level)

- Phase 2A: Introduce lock-free queue primitives between stages.
- Phase 2B: Implement per-stage worker loops with thread pools and proper lifecycle management.
- Phase 2C: Wire the pipeline to use the existing Phase 1 ONNX session pool for Inference.
- Phase 2D: Integrate health checks and basic instrumentation.
- Phase 2E: Build and verify through end-to-end tests and performance benchmarks.

Risks & Mitigations

- Complexity: add clear abstractions, avoid over-engineering; iterate with small increments.
- Backpressure tuning: adjust queue sizes to balance latency and memory.
- Ensuring deterministic behavior: add deterministic scheduling where needed for reproducibility.

Appendix: Integration Points

- Inference: src/inference.h/.cpp (session pool, RunSession, WarmUpSession)
- VideoController: src/VideoController.h/.cpp (pipeline glue, worker threads)
- Main: main.cpp (application lifecycle)
- Docs: this PHASE2_SKETCH.md document

Diagram: Phase 2 Architecture

```
┌ Phase 2 Architecture ───────────────────────────────────┐
│                                                     │
│ VideoController                       UI            │
│ ┌───────────┐                                      │
│ │ Camera    │                                      │
│ │ Worker(s) │                                      │
│ │ 2 threads │                                      │
│ └───────────┘                                      │
│       │                                             │
│       ▼                                             │
│ [FrameQueue]  Lock-free, bounded                    │
│       ▼                                             │
│ [Preprocess] 4 threads                              │
│       │                                             │
│       ▼                                             │
│ [PreprocessQueue]                                   │
│       ▼                                             │
│ [Inference] 2–3 threads (Pool)                     │
│   ├─ Session Pool: S1, S2, ... SN                   │
│   └─ Round-robin dispatch                           │
│       ▼                                             │
│ [InferenceQueue]                                    │
│       ▼                                             │
│ [Postprocess] 2 threads                            │
│       ▼                                             │
│ [PostprocessQueue]                                 │
│       ▼                                             │
│ Display/Rendering (UI)                               │
└─────────────────────────────────────────────────────┘
```
