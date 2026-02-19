# Product Requirements Document (PRD)

**Project Name**: QtOpenCVCamera (YOLOApp)
**Version**: 0.2
**Status**: Functional Prototype

## 1. Introduction
The **QtOpenCVCamera** is a high-performance desktop application designed to demonstrate real-time object detection using State-of-the-Art (SOTA) AI models within a modern Qt Quick interface. The application features a fully asynchronous, multi-threaded architecture optimized for maximum throughput on CPU hardware.

## 2. Goals
- **High Performance**: Achieve 30+ FPS camera capture with decoupled inference running at model-limited speed (~10-20 FPS on CPU for YOLOv8n).
- **Responsive UI**: Ensure the UI never freezes by isolating capture, inference, and monitoring into separate threads.
- **Modern Stack**: Showcase integration of C++17, Qt 6.8, OpenCV 4.x, and ONNX Runtime.
- **Maintainability**: Clean architecture separating UI (QML), control logic (VideoController), data models (DetectionListModel), and AI engine (YOLO_V8).
- **Observability**: Real-time performance metrics (camera FPS, inference FPS, per-phase timing, CPU/RAM usage).

## 3. User Stories
- **As a User**, I want to see a live feed from my webcam so I can observe my environment.
- **As a User**, I want the application to highlight objects (like 'Person', 'Car', 'Dog') with color-coded bounding boxes and labels so I can identify them.
- **As a User**, I want to see inference timing and FPS metrics so I can understand the application's performance.
- **As a User**, I want to monitor system performance (CPU/RAM usage) to ensure the app isn't consuming excessive resources.
- **As a Developer**, I want the AI model to be interchangeable (ONNX format) so I can upgrade detection capabilities by swapping model files.
- **As a Developer**, I want the capture and inference pipelines to be decoupled so I can optimize each independently.

## 4. Functional Requirements
1. **Video Capture**: Support standard USB webcams via OpenCV (DSHOW backend, MJPG codec, 640×480).
2. **Object Detection**:
   - Model: YOLOv8 Nano (or higher, configurable via `DL_INIT_PARAM`).
   - Classes: MS COCO (80 classes loaded from `classes.txt`).
   - Visualization: Color-coded bounding boxes (Scene Graph) + class labels + confidence scores (QML overlays).
   - Frame-drop logic: Gracefully skip frames when inference is slower than capture.
3. **Performance Overlay (HUD)**:
   - Camera FPS (capture thread throughput).
   - Inference FPS (smoothed model throughput).
   - Pre-process, inference, and post-process timing in milliseconds (3 decimal places).
   - CPU usage and process RAM usage.
4. **UI/UX**:
   - Windowed application (800×600 default, dark theme).
   - Live video preview with bounding box overlay.
   - "Close App" button.

## 5. Non-Functional Requirements
- **Target OS**: Windows 10/11 (Primary), Linux (Secondary — SystemMonitor supports it).
- **Startup Time**: < 3 seconds (thread startup deferred by 500ms to prevent UI freeze).
- **Latency**: End-to-end detection latency determined by model speed (~50-100ms for YOLOv8n on CPU).
- **Stability**: No memory leaks — multi-buffer frame pool, reusable blob buffers, RAII throughout.
- **Thread Safety**: Atomic flags for worker state, `QueuedConnection` for cross-thread signals, `compare_exchange_strong` for frame-drop logic.

## 6. Constraints
- Must run on CPU (CUDA optional via `DL_INIT_PARAM.cudaEnable`).
- Must use CMake build system.
- Inference engine code is derived from Ultralytics (AGPL-3.0 licensed).
