# Definition of Done (DoD)

For a feature or task to be considered "Done" in the **QtOpenCVCamera** project, it must meet the following criteria:

## 1. Code Quality
- [ ] Code compiles without warnings on MSVC/GCC.
- [ ] Follows C++17 standards (smart pointers, no raw `new`/`delete` unless wrapped).
- [ ] Formatted using `clang-format` (Google or LLVM style).
- [ ] No hardcoded paths (use configuration or relative paths).

## 2. Architecture
- [ ] QML and C++ are decoupled (interaction only via Signals/Slots/Properties).
- [ ] No blocking operations in the Main Thread.
- [ ] Classes have single responsibility principle applied.

## 3. Testing
- [ ] Feature verified manually on Windows.
- [ ] No regressions in FPS (Frame Rate) for the baseline config.
- [ ] Memory usage is stable (no leaks detected via Task Manager or CRT Debug heap).

## 4. Documentation
- [ ] Public API methods (C++ headers) are commented.
- [ ] Architecture diagrams updated if the structure changed.
- [ ] Major changes logged in `docs/` (e.g., tech stack updates).

## 5. Build & Deployment
- [ ] CMake config updated and verifies successful build.
- [ ] Required DLLs (ONNX Runtime, OpenCV) are copied to the build output directory definition.
