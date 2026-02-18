# Tech Stack

## Core Technologies
- **Language**: C++17 (Application Logic), QML (User Interface)
- **Framework**: Qt 6.8 (Multimedia, Quick, Core)
- **Computer Vision**: OpenCV 4.x (Video Capture, Image Pre-processing)
- **AI Inference**: ONNX Runtime (v1.16+ recommended)
- **Model**: YOLOv8 (Nano version recommended for realtime CPU performance)

## Development Environment
- **OS**: Windows 10/11
- **Compiler**: MSVC 2019+ or MinGW w/ GCC 8+
- **Build System**: CMake 3.16+
- **IDE**: VS Code, Qt Creator, or Visual Studio

## Architecture
- **Pattern**: Signal/Slot based hybrid QML/C++.
- **Threading**:
    - **Main Thread**: QML UI rendering and event handling.
    - **Worker Thread**: Video capture (`cv::VideoCapture`) and AI Inference (`Ort::Session::Run`).
- **Memory Management**: RAII for C++ resources, parent-child ownership for Qt objects.

## Key Libraries & Dependencies
| Library | Purpose | Integration |
| :--- | :--- | :--- |
| **Qt Multimedia** | Displaying video frames in QML (`QVideoSink`) | `find_package(Qt6 COMPONENTS Multimedia)` |
| **Qt Quick** | Hardware-accelerated UI | `find_package(Qt6 COMPONENTS Quick)` |
| **OpenCV** | Reading frames from webcam | `find_package(OpenCV)` |
| **ONNX Runtime** | Running the YOLOv8 neural network | `target_link_libraries(... onnxruntime)` |
| **PDH / PSAPI** | detailed system monitoring (Windows only) | `target_link_libraries(... pdh psapi)` |
