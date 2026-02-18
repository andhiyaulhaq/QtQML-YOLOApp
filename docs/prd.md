# Product Requirements Document (PRD)

**Project Name**: QtOpenCVCamera (YOLOApp)
**Version**: 0.1
**Status**: Prototype

## 1. Introduction
The **QtOpenCVCamera** is a high-performance desktop application designed to demonstrate real-time object detection using State-of-the-Art (SOTA) AI models within a modern Qt Quick interface.

## 2. Goals
- **High Performance**: Achieve 30+ FPS detection on standard CPU hardware.
- **Responsive UI**: Ensure the UI never freezes during heavy inference loads.
- **Modern Stack**: Showcase integration of C++17, Qt 6, and ONNX Runtime.
- **Maintainability**: Clean architecture separating UI, Logic, and AI.

## 3. User Stories
- **As a User**, I want to see a live feed from my webcam so I can observe my environment.
- **As a User**, I want the application to highlight objects (like 'Person', 'Car', 'Dog') with bounding boxes so I can identify them.
- **As a User**, I want to monitor system performance (CPU/RAM usage) to ensure the app isn't hogging resources.
- **As a Developer**, I want the AI model to be interchangeable (ONNX format) so I can upgrade detection capabilities easily.

## 4. Functional Requirements
1. **Video Capture**: Support standard USB webcams via OpenCV.
2. **Object Detection**:
   - Model: YOLOv8 Nano (or higher, configurable).
   - Classes: MS COCO (80 classes).
   - Visualization: Bounding boxes + Class Labels + Confidence scores.
3. **UI/UX**:
   - Windowed application (resizable).
   - "Close App" button.
   - Live video preview.

## 5. Non-Functional Requirements
- **Target OS**: Windows 10/11 (Primary), Linux (Secondary).
- **Startup Time**: < 3 seconds.
- **Latency**: End-to-end latency < 100ms.
- **Stability**: No memory leaks over 24 hours of runtime.

## 6. Constraints
- Must run on CPU (CUDA optional but not required for baseline).
- Must use CMake build system.
- Must not use GPL-licensed modules if proprietary distribution is planned (strictly adhere to AGPL/MIT/Apache compatibilities).
