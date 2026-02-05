Phase 1 Sketch: ONNX Threading + Session Pool (Merged Phase 0)

Purpose

- Document how the current Phase 1 work is wired to use multiple ONNX Runtime sessions and how the loads are balanced across them to improve throughput on multi-core CPUs.

What Phase 1 comprises

- Reusing the existing ONNX Runtime threading setup (intraOp and interOp thread counts) and graph optimizations.
- Introducing a pool of Ort::Session instances (sessionPool) and a round-robin dispatcher to distribute inferences.
- Per-session warm-up to reduce cold-start latency and stabilize performance.
- Cleanup of all pooled sessions on shutdown.

Key classes and data paths

- YOLO_V8 (src/inference.h/.cpp)
  - Private: std::vector<Ort::Session\*> m_sessionPool; std::atomic<size_t> m_sessionIndex;
  - In CreateSession: initialize N sessions based on sessionPoolSize; push into m_sessionPool
  - RunSession / TensorProcess: select session via round-robin, invoke Run on the chosen session
  - WarmUpSession: perform warm-up on all pooled sessions
  - Destructor: clean up all pooled sessions

- Shared inputs/outputs:
  - inputNodeNames and outputNodeNames are built from the first session and reused for all pool members

How data flows (end-to-end)

- Capture → PreProcess → Blob creation → Input tensor creation → dispatch to a pool member (session) → Run → Output tensor processing → PostProcess → Drawing/Display
- Dispatch uses a simple round-robin with an atomic index to pick the next session.

Configuration knobs (example)

- sessionPoolSize: number of Ort::Session instances in the pool (2–4 is a typical starting point)
- intraOpNumThreads: as in hardware_concurrency() (e.g., 8)
- interOpNumThreads: keep at 1 to minimize contention
- modelPath, imgSize, rectConfidenceThreshold, iouThreshold, cudaEnable: unchanged from existing config

Validation steps

- Build with sessionPoolSize > 1 and observe:
  - Higher CPU utilization across cores
  - Throughput improvement (FPS) vs single-session baseline
  - Similar or reduced tail latency per inference
- Validate warm-up runs complete for all pool members (log messages help)
- Check for memory leaks by running long sessions and reviewing memory usage

Potential pitfalls & mitigations

- Memory footprint grows with pool size; start small (2–4) and scale up if needed
- IO name consistency: reuse inputNodeNames/outputNodeNames across sessions; ensure model IO remains identical
- Synchronization: the atomic index guards against data races in session selection

Next steps (if you want to extend Phase 1)

- Add a small test harness to simulate bursty inputs and measure max sustained FPS
- Introduce a lightweight health check for sessions and an optimistic pool resize strategy

Notes

- This sketch focuses on the mechanics of Phase 1. Phase 2+ introduces a full multi-stage pipeline, memory pools, and advanced synchronization guarantees.

Diagram: Current Phase 1 Architecture

```
┌───────────────────────────────────────────────────────────┐
│                 Working Phase 1 System                    │
├───────────────────────────────────────────────────────────┤
│ VideoController                                           │
│ ┌──────────┐                                              │
│ │ Camera   │                                              │
│ │ Worker   │                                              │
│ │ (single  │                                              │
│ │  thread) │                                              │
│ └──────────┘                                              │
│     │                                                     │
│     ▼                                                     │
│ YOLO_V8 Session Pool (N sessions)                         │
│ - m_sessionPool: [S1, S2, ..., SN]                        │
│ - Round-robin dispatcher                                  │
│ - Warm-up all pooled sessions                             │
│                                                           │
│     ▼                                                     │
│ Inference results -> Post-process -> Draw -> Display      │
│                                                           │
│     ▼                                                     │
│ QVideoSink / UI (Video display)                           │
└───────────────────────────────────────────────────────────┘
```
