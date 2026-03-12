# QtOpenCVCamera (YOLOApp)

**QtOpenCVCamera** is a high-performance desktop application that demonstrates real-time object detection, image segmentation, and pose estimation using **YOLOv8**. It supports dual inference runtimes—**OpenVINO** (default) and **ONNX Runtime**—integrated with a modern **Qt Quick (QML)** user interface.

![Modern C++](https://img.shields.io/badge/C++-17-blue.svg)
![Qt](https://img.shields.io/badge/Qt-6.8.3-green.svg)
![OpenVINO](https://img.shields.io/badge/OpenVINO-2024-blue.svg)
![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-orange.svg)

## 🚀 Features

- **Multi-Task Support**: Switch between **Object Detection**, **Pose Estimation**, and **Image Segmentation** at runtime.
- **Dual Inference Backends**: Choose between **OpenVINO** (optimized for Intel hardware) and **ONNX Runtime** (CPU/CUDA) via the UI.
- **High-Performance Architecture**: Decoupled background threads for capture, inference, and system monitoring ensure a smooth 60+ FPS UI.
- **Hardware-Accelerated Rendering**: Custom Scene Graph implementation (`DetectionOverlayItem`) for low-latency visualization of masks, skeletons, and boxes.
- **Real-time Metrics**: Live tracking of inference timing (pre/inf/post), FPS, and system resource (CPU/RAM) usage.

## 🛠️ Prerequisites

To build this project on Windows, you need:

1.  **C++ Compiler**: [MSVC 2022](https://visualstudio.microsoft.com/vs/community/) (v143).
2.  **Qt 6.8.3 SDK**: Specifically the **MSVC 2022 64-bit** component.
3.  **OpenVINO Toolkit**: Version 2024.x (installed in `C:/intel/openvino_toolkit`).
4.  **OpenCV**: Version 4.12.0 (installed in `C:/opencv`).
5.  **ONNX Runtime**: Version 1.16+ (installed in `C:/onnxruntime`).
6.  **CMake**: Version 3.16 or higher.

## 📦 Build Instructions (MSVC)

```bash
# 1. Clone the repository
git clone https://github.com/andhiyaulhaq/QtQML-YOLOApp.git
cd QtQML-YOLOApp

# 2. Configure via CMake using MSVC 2022
# Note: Ensure CMAKE_PREFIX_PATH points to your MSVC Qt installation
cmake -G "Visual Studio 17 2022" -B build -S . -DCMAKE_PREFIX_PATH="C:/Qt/6.8.3/msvc2022_64"

# 3. Build the application (Release)
cmake --build build --config Release --target appCamera
```

## 🚀 Deployment & Running

To run the application outside of an IDE, you must deploy the Qt dependencies:

```bash
# 1. Run windeployqt to copy Qt DLLs and plugins
C:/Qt/6.8.3/msvc2022_64/bin/windeployqt.exe --release --qmldir content --qmldir . build/Release/appCamera.exe

# 2. Run the application
cd build/Release
./appCamera.exe
```

*Note: The build system automatically copies OpenCV, OpenVINO, and ONNX Runtime DLLs to the Release folder after compilation.*

## 🏗️ Architecture

The project uses a **Strategy Pattern** for inference backends and a **Multi-Threaded C++/QML Bridge**:

- **Frontend (QML)**: Modern UI with a sidebar for task and runtime switching.
- **Backend (C++)**:
    - `VideoController`: QML Bridge and thread orchestrator.
    - `IInferenceBackend`: Abstract interface for `OpenVinoBackend` and `OnnxRuntimeBackend`.
    - `YOLO`: Main inference engine delegating to the selected backend.
    - `DetectionOverlayItem`: High-performance custom item for rendering AI results.

## 📄 License

This project is licensed under the [MIT License](LICENSE).
