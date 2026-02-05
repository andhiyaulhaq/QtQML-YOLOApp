Phase 4 Sketch: Thread Safety & Synchronization

Purpose
- Harden the system against race conditions and ensure robust parallelism with lock-free and atomic primitives.

Assumptions
- hot paths require lock-free approaches; only non-critical sections may tolerate mutexes.
- Phase 1 pool and Phase 2 pipeline are present; Phase 4 builds on top of them.

Deliverables
- Lock-free queues for core data paths (frames, tensors, results).
- Thread-local resources for per-session or per-stage state.
- Atomic metrics counters and health signals.
- Centralized error propagation and health checks.
- Concurrency tests and sanitizer-enabled builds.

Implementation Plan
- Phase 4A: Replace remaining mutex-protected paths with lock-free structures where feasible.
- Phase 4B: Introduce thread-local storage for per-session and per-stage resources to minimize contention.
- Phase 4C: Add atomic counters for per-stage throughput, latency, and error states.
- Phase 4D: Implement health checks and a simple watchdog mechanism for pipeline health.
- Phase 4E: Add concurrency tests (data race tests, stress tests) and enable sanitizers in CI.

Risks & Mitigations
- Complexity risk: incrementally transition; keep fallback paths for safety.
- Atomic bugs: use well-known lock-free patterns and formal verification where possible.
- Debug difficulty: add logging gates and structured tracing to diagnose issues.

Validation & Metrics
- Race-condition detection via sanitizers (AddressSanitizer, ThreadSanitizer).
- Per-stage latency histograms and flow-based tracing.
- Confirm no deadlocks or livelocks under bursty workloads.

Integration Points
- Inference: src/inference.h/.cpp
- VideoController: src/VideoController.*
- Phase 2: memory and queue abstractions

Diagram (ASCII)
```
┌ Phase 4: Thread Safety ────────────────────────────────────┐
│                                                       │
│  Lock-free data paths across stages (Frame, Blob, …) │
│  Thread-local storage for per-thread resources          │
│  Atomic metrics: throughput, latency, errors            │
│  Health checks and error propagation                    │
└──────────────────────────────────────────────────────────┘
```

End of Sketch
