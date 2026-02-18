# QtOpenCVCamera (YOLOApp)

**QtOpenCVCamera** is a high-performance desktop application that demonstrates real-time object detection using **YOLOv8** (via ONNX Runtime) integrated with a modern **Qt Quick (QML)** user interface.

![Modern C++](https://img.shields.io/badge/C++-17-blue.svg)
![Qt](https://img.shields.io/badge/Qt-6.8-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red.svg)
![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-orange.svg)

## 🚀 Features

- **Live Object Detection**: Real-time bounding box visualization for 80+ object classes (COCO dataset).
- **High Performance**: Multithreaded architecture ensures a smooth 60 FPS UI while processing heavy AI inference in the background.
- **Modern UI**: Sleek, hardware-accelerated interface built with Qt Quick and QML.
- **System Monitoring**: Built-in CPU and RAM usage tracking to ensure efficiency.
- **Cross-Platform Ready**: Designed with CMake for easy porting to Linux (primary target: Windows 10/11).

## 🛠️ Prerequisites

To build this project, you need the following tools installed:

1.  **C++ Compiler**: MSVC 2019+ (Windows) or GCC 8+ (Linux).
2.  **CMake**: Version 3.16 or higher.
3.  **Qt 6 SDK**: Including `Qt Multimedia` and `Qt Quick` modules.
4.  **OpenCV**: Version 4.x (built with GStreamer support recommended on Linux).
5.  **ONNX Runtime**: Version 1.16+ (Shared library).

## 📦 Build Instructions

```bash
# 1. Clone the repository
git clone https://github.com/andhiyaulhaq/QtQML-YOLOApp.git
cd QtQML-YOLOApp

# 2. Create a build directory
mkdir build
cd build

# 3. Configure via CMake (Adjust paths to your OpenCV/ONNX installations)
cmake .. -DOpenCV_DIR="C:/opencv/build" -DOnnxRuntime_ROOT="C:/onnxruntime"

# 4. Build the application
cmake --build . --config Release

# 5. Run
./Release/appCamera.exe
```

## 🏗️ Architecture

The project follows a **Hybrid C++/QML Architecture**:

-   **Frontend (QML)**: Handles all UI rendering, animations, and user inputs. It communicates with the C++ backend via Qt's Signal/Slot mechanism.
-   **Backend (C++)**:
    -   `VideoController`: Manages the camera lifecycle and the background worker thread.
    -   `CameraWorker`: Runs the main capture loop. It fetches frames from OpenCV, runs inference, and pushes results to the UI.
    -   `YOLO_V8`: A wrapper around ONNX Runtime for performing efficient inference.

For more details, check the **[Documentation Suite](docs/)**:
-   [Project Structure](docs/project-structure.md)
-   [Tech Stack](docs/tech-stack.md)
-   [Design Specifications](docs/design-spec.md)

## 🤝 Contributing

Contributions are welcome! Please see the `CONTRIBUTING.md` (coming soon) for details.

## 📄 License

This project is licensed under the [MIT License](LICENSE).
