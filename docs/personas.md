# User Personas

## 1. The Security Operator - "Sam"
- **Background**: Works in a security room monitoring multiple feeds.
- **Goals**: Needs reliable, continuous object detection to alert on specific events (e.g., "Person" in restricted area).
- **Frustrations**: Laggy video feeds, confusing interfaces, high CPU usage slowing down other monitoring tools.
- **Needs**: High stability, low resource footprint, clear visual indicators.

## 2. The AI R&D Developer - "Alex"
- **Background**: Computer Vision engineer prototyping new models.
- **Goals**: Wants a solid "host" application to test custom trained YOLO models without rewriting UI code.
- **Frustrations**: Hard-coded model paths, difficult build systems, lack of debug info.
- **Needs**: Clean code structure, passing custom ONNX files, visualization of inference time.

## 3. The Qt Enthusiast - "Jordan"
- **Background**: C++ developer looking to learn QML and modern C++.
- **Goals**: Wants to see "Best Practices" for integrating heavy C++ backend work with QML.
- **Frustrations**: Examples that do everything in `main.cpp` or block the GUI thread.
- **Needs**: Proper threading examples, signal/slot usage, modern CMake setup.
