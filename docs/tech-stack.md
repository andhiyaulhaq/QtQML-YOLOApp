# Tech Stack

## Core Technologies
- **Language**: C++17 (Application Logic), QML (User Interface)
- **Framework**: Qt 6.8 (Quick, Multimedia, Core)
- **Computer Vision**: OpenCV 4.x (Video Capture, Image Preprocessing, Color Conversion)
- **AI Inference**: ONNX Runtime (v1.16+ recommended)
- **Model**: YOLOv8 Nano (`yolov8n.onnx`) for real-time CPU performance

## Development Environment
- **OS**: Windows 10/11 (Primary Target)
- **Compiler**: MSVC 2019+ or MinGW with GCC 8+
- **Build System**: CMake 3.16+
- **IDE**: VS Code, Qt Creator, or Visual Studio

## Architecture
- **Pattern**: Signal/Slot based hybrid QML/C++ with `QML_ELEMENT` registration.
- **Threading** (4 threads total):

| Thread | Class | Priority | Responsibility |
| :--- | :--- | :--- | :--- |
| **Main** | QML Engine | Normal | UI rendering, event handling, signal routing |
| **Capture** | `CaptureWorker` | Normal | Camera capture loop, frame delivery to UI + inference |
| **Inference** | `InferenceWorker` | High | YOLO model loading, inference execution, frame-drop logic |
| **Monitor** | `SystemMonitor` | Low | CPU/RAM polling via platform-native APIs |

- **Rendering**: Hybrid approach — `BoundingBoxItem` (C++ Scene Graph) for box geometry + QML `Repeater` for text labels.
- **Memory Management**: RAII for C++ resources, parent-child ownership for Qt objects, multi-buffer frame pool to minimize allocations.
- **Data Bridge**: `DetectionListModel` (`QAbstractListModel`) + `Detection` (`Q_GADGET`) struct for efficient C++ → QML data binding.

## Key Libraries & Dependencies
| Library | Purpose | Integration |
| :--- | :--- | :--- |
| **Qt Quick** | Hardware-accelerated UI, Scene Graph rendering | `find_package(Qt6 COMPONENTS Quick)` |
| **Qt Multimedia** | Displaying video frames in QML (`QVideoSink`, `QVideoFrame`) | `find_package(Qt6 COMPONENTS Multimedia)` |
| **OpenCV** | Camera capture (`VideoCapture`), color conversion, image resizing | `find_package(OpenCV)` |
| **ONNX Runtime** | Running the YOLOv8 neural network model | `target_link_libraries(... onnxruntime onnxruntime_providers_shared)` |
| **PDH / PSAPI** | Windows-specific CPU and memory monitoring | `target_link_libraries(... pdh psapi)` (Windows only) |

## Build Optimizations
- **Release Flags (MSVC)**: `/O2 /Ob2 /Oi /Ot /Oy /GL` with `/LTCG` linker optimization.
- **Release Flags (GCC/MinGW)**: `-O3 -march=native -ffast-math`.
- **ONNX Threading**: Capped `intraOpNumThreads` at `min(4, hardware_concurrency/2)` for nano model efficiency.
