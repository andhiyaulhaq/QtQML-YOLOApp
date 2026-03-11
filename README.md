# QtOpenCVCamera (YOLOApp)

**QtOpenCVCamera** is a high-performance desktop application that demonstrates real-time object detection using **YOLOv8** (via ONNX Runtime) integrated with a modern **Qt Quick (QML)** user interface. The application features a fully asynchronous, multi-threaded architecture that decouples video capture, AI inference, and UI rendering for maximum throughput.

![Modern C++](https://img.shields.io/badge/C++-17-blue.svg)
![Qt](https://img.shields.io/badge/Qt-6.8-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red.svg)
![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-orange.svg)

## 🚀 Features

- **Live Object Detection**: Real-time bounding box visualization with class labels and confidence scores for 80 object classes (COCO dataset).
- **High-Performance Multi-Threaded Architecture**: Three dedicated background threads (capture, inference, monitoring) ensure a smooth 60 FPS UI that never blocks.
- **Hardware-Accelerated Rendering**: Bounding boxes are rendered via Qt's Scene Graph (`DetectionOverlayItem`) for minimal GPU overhead, with QML-based text labels.
- **Inference Timing**: Displays pre-process, inference, and post-process durations (ms precision) alongside camera and inference FPS.
- **System Monitoring**: Built-in CPU and RAM usage tracking via platform-native APIs (PDH/PSAPI on Windows).
- **Memory-Optimized Pipeline**: Multi-buffer frame pool, frame-drop logic, and reusable blob buffers minimize allocations in the hot path.

## 🛠️ Prerequisites

To build this project, you need the following tools installed:

1.  **C++ Compiler**: MSVC 2019+ (Windows) or GCC 8+ (Linux).
2.  **CMake**: Version 3.16 or higher.
3.  **Qt 6 SDK**: Including `Qt Multimedia` and `Qt Quick` modules.
4.  **OpenCV**: Version 4.x.
5.  **ONNX Runtime**: Version 1.16+ (Shared library).

## 📦 Build Instructions

```bash
# 1. Clone the repository
git clone https://github.com/andhiyaulhaq/QtQML-YOLOApp.git
cd QtQML-YOLOApp

# 2. Create a build directory
mkdir build
cd build

# 3. Configure via CMake (MinGW Makefiles)
# Note: Ensure OpenCV and ONNX Runtime are in your PATH or provide -DOpenCV_DIR and -DOnnxRuntime_ROOT
cmake -G "MinGW Makefiles" ..

# 4. Build the application
mingw32-make

# 5. Run
./appCamera.exe
```

## 🏗️ Architecture

The project follows a **Multi-Threaded C++/QML Architecture** with three dedicated background threads:

```
┌─────────────────────────────────────────────────────────────┐
│  Main Thread (GUI)                                          │
│  ┌──────────────────────────────────────────────┐           │
│  │  QML UI (Main.qml)                           │           │
│  │  ├── VideoOutput (raw camera feed)           │           │
│  │  ├── DetectionOverlayItem (Scene Graph overlay)   │           │
│  │  │   └── Repeater (text labels per box)      │           │
│  │  └── Performance Overlay (FPS, timing, CPU)  │           │
│  └────────────────┬─────────────────────────────┘           │
│                   │ Signals/Slots                            │
│  ┌────────────────┴─────────────────────────────┐           │
│  │  VideoController (C++ ↔ QML Bridge)          │           │
│  │  ├── DetectionListModel (QAbstractListModel) │           │
│  │  └── Thread lifecycle management             │           │
│  └────────────────┬─────────────────────────────┘           │
└───────────────────┼─────────────────────────────────────────┘
                    │ QueuedConnection
  ┌─────────────────┼──────────────────────────────────────┐
  │                 ▼                                      │
  │  ┌──────────────────────┐    ┌─────────────────────┐   │
  │  │  Capture Thread      │───▶│  Inference Thread   │   │
  │  │  (CaptureWorker)     │    │  (InferenceWorker)  │   │
  │  │  ├ cv::VideoCapture  │    │  ├ YOLO_V8 engine   │   │
  │  │  ├ Frame Pool [3]    │    │  ├ Frame drop logic │   │
  │  │  └ FPS calculation   │    │  └ Timing metrics   │   │
  │  └──────────────────────┘    └─────────────────────┘   │
  │                                                        │
  │  ┌──────────────────────┐                              │
  │  │  Monitor Thread      │                              │
  │  │  (SystemMonitor)     │                              │
  │  │  └ CPU / RAM stats   │                              │
  │  └──────────────────────┘                              │
  └────────────────────────────────────────────────────────┘
```

-   **Frontend (QML)**: Renders the camera feed via `VideoOutput`, overlays bounding boxes using a custom Scene Graph item (`DetectionOverlayItem`), and displays labels via a QML `Repeater`.
-   **Backend (C++)**:
    -   `VideoController`: Orchestrates thread lifecycle, bridges QML ↔ C++ via properties/signals, and owns the `DetectionListModel`.
    -   `CaptureWorker`: Runs the camera capture loop with a 3-frame ring buffer, pushes raw frames to `QVideoSink` and emits them to the inference pipeline.
    -   `InferenceWorker`: Loads the YOLO model, processes frames with frame-drop logic, and emits detection results + timing back to the main thread.
    -   `YOLO_V8`: ONNX Runtime wrapper with session pooling, letterbox preprocessing, and NMS postprocessing.
    -   `SystemMonitor`: Platform-native CPU/RAM monitoring on a low-priority thread.

For more details, check the **[Documentation Suite](docs/)**:
-   [Project Structure](docs/project-structure.md)
-   [Tech Stack](docs/tech-stack.md)
-   [Design Specifications](docs/design-spec.md)
-   [Product Requirements](docs/prd.md)

## 🤝 Contributing

Contributions are welcome! Please see the `CONTRIBUTING.md` (coming soon) for details.

## 📄 License

This project is licensed under the [MIT License](LICENSE).
