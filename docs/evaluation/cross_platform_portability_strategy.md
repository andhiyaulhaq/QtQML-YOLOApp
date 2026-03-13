# Cross-Platform Portability Strategy: YOLOApp
**Date:** 2026-03-13 16:57
**Status:** Initial Proposal

## 1. Executive Summary
The YOLOApp is currently a Windows-centric C++ application using Qt 6, OpenCV, and multiple inference runtimes (ONNX Runtime, OpenVINO). While the core technologies are cross-platform, the current implementation relies on Windows-specific APIs for camera capture, environment configuration, and DLL management, alongside hardcoded setup paths in CMake.

This document outlines the strategy to transform YOLOApp into a truly portable application capable of running on **Windows, Linux (x86/ARM), and macOS (Intel/Apple Silicon)**.

---

## 2. Dependency Assessment

| Dependency | Windows | Linux | macOS | Mitigation Strategy |
| :--- | :---: | :---: | :---: | :--- |
| **Qt 6 / QML** | ✅ | ✅ | ✅ | Use standard Qt layouts and styles. |
| **OpenCV** | ✅ | ✅ | ✅ | Use `find_package` instead of hardcoded paths. |
| **ONNX Runtime** | ✅ | ✅ | ✅ | Use CPU/CoreML/DirectML providers conditionally. |
| **OpenVINO** | ✅ | ✅ | ⚠️ | **Unsupported on Apple Silicon.** Make optional via CMake. |
| **Pdh / Psapi** | ✅ | ❌ | ❌ | Existing `SystemMonitor` already has Linux/Mac branches. |

---

## 3. Key Technical Challenges

### A. The "OpenVINO" macOS Blocker
OpenVINO is highly optimized for Intel but does not support Apple Silicon (M-series) natively for inference. 
*   **Solution:** Introduce a CMake option `ENABLE_OPENVINO`. If OFF (default on macOS ARM), the `OpenVinoBackend` will be stubbed or excluded from the build, and the UI will disable the OpenVINO runtime option.

### B. Hardcoded CMake Paths
Current `CMakeLists.txt` uses absolute paths like `C:/opencv/build`.
*   **Solution:** Implement `find_package()` for all dependencies. Use environment variables (e.g., `OpenCV_DIR`, `ONNXRUNTIME_ROOT`) to allow developers to specify dependency locations on different systems without modifying code.

### C. Camera Capture Backend
`VideoController.cpp` explicitly requests `cv::CAP_DSHOW` (Windows DirectShow).
*   **Solution:** Use `cv::CAP_ANY` or platform-specific guards. On Linux, `V4L2` is preferred; on macOS, `AVFOUNDATION`.

### D. DLL / Shared Library Management
The Windows `.bat` approach and `SetDllDirectory` are non-portable.
*   **Solution:** 
    *   **Linux:** Rely on `RPATH` settings in CMake and standard `.so` placement in `/usr/lib` or alongside the executable.
    *   **macOS:** Use `@executable_path` and `App Bundles` (.app) which contain libraries in `Frameworks/`.

---

## 4. Proposed Implementation Roadmap

### Phase 1: Infrastructure Cleanup (Immediate)
1.  **Refactor CMakeLists.txt:** 
    *   Remove all hardcoded `C:/...` paths.
    *   Add `option(ENABLE_OPENVINO "Build with OpenVINO support" ON)`.
    *   Wrap OpenVINO-specific code in `if(ENABLE_OPENVINO)` blocks.
2.  **Path Abstraction:** 
    *   Create a `PathUtils` helper to handle assets/models path resolution using `QCoreApplication::applicationDirPath()`.

### Phase 2: Code Decoupling
1.  **Inference Backends:**
    *   Add `#ifdef ENABLE_OPENVINO` guards in `YoloPipeline.cpp` and `VideoController.cpp`.
2.  **OnnxRuntime Path Handling:**
    *   Replace Windows-only `MultiByteToWideChar` logic in `OnnxRuntimeBackend.cpp` with a cross-platform string converter or direct `char*` paths for Linux/Mac.
3.  **Capture Optimization:**
    *   Abstract `m_capture.open(0, cv::CAP_DSHOW)` to a method that picks the best backend per OS.

### Phase 3: Platform Porting & Testing
1.  **Linux Port:**
    *   Verify `SystemMonitor` `/proc/stat` logic.
    *   Set up AppImage or Flatpak packaging.
2.  **macOS Port:**
    *   Implement DMG bundling.
    *   Test ONNX Runtime with CoreML execution provider for hardware acceleration.

---

## 5. Deployment Strategy

*   **Windows:** Keep the current Release structure but replace `.bat` with a proper installer or `windeployqt` handled by CMake.
*   **Linux:** Provide a `.sh` runner or a standalone **AppImage**.
*   **macOS:** Distribute as a standard **.app bundle** in a **.dmg**.

---

> [!IMPORTANT]
> To maintain the high-performance "Pro" feel, hardware acceleration is critical. We must ensure ONNX Runtime uses **DirectML** on Windows, **OpenVINO/TensorRT** on Linux, and **CoreML** on macOS.
