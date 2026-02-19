# Definition of Done (DoD)

For a feature or task to be considered "Done" in the **QtOpenCVCamera** project, it must meet the following criteria:

## 1. Code Quality
- [ ] Code compiles without warnings on MSVC/GCC with `-O3` optimizations enabled.
- [ ] Follows C++17 standards (smart pointers, no raw `new`/`delete` unless wrapped by RAII or Qt parent ownership).
- [ ] Formatted using `clang-format` (Google or LLVM style).
- [ ] No hardcoded paths in application logic (use relative paths or configuration constants like `AppConfig`).

## 2. Architecture
- [ ] QML and C++ are decoupled (interaction only via `Q_PROPERTY`, signals/slots, and `QML_ELEMENT`).
- [ ] No blocking operations in the Main Thread (capture, inference, and monitoring run on dedicated threads).
- [ ] Cross-thread communication uses `QueuedConnection` (or `DirectConnection` only for stop signals where safe).
- [ ] Thread-safe state management (atomic flags for worker running/processing states).
- [ ] Classes follow Single Responsibility Principle.

## 3. Performance
- [ ] Camera FPS remains at capture rate (~30 FPS) after changes.
- [ ] Inference pipeline does not accumulate queued frames (frame-drop logic functional).
- [ ] No new memory allocations in the hot path (use pre-allocated buffers where possible).
- [ ] Memory usage is stable (no leaks detected via Task Manager or CRT Debug heap).

## 4. Testing
- [ ] Feature verified manually on Windows (MinGW Makefiles build).
- [ ] Bounding boxes, labels, and confidence scores render correctly.
- [ ] Performance HUD displays accurate timing and FPS values.
- [ ] Application starts and shuts down cleanly without hangs.

## 5. Documentation
- [ ] Public API methods (C++ headers) are commented.
- [ ] Architecture diagrams updated if the threading model or component structure changed.
- [ ] Major changes logged in `docs/` (design-spec, tech-stack, project-structure as appropriate).

## 6. Build & Deployment
- [ ] CMake config updated and successfully builds with `cmake -G "MinGW Makefiles"`.
- [ ] Required DLLs (ONNX Runtime, OpenCV) are copied to the build output directory via `add_custom_command`.
- [ ] Model files (`inference/`) are copied to the build output directory.
