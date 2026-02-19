# User Personas

## 1. The Security Operator — "Sam"
- **Background**: Works in a security room monitoring multiple feeds.
- **Goals**: Needs reliable, continuous object detection to alert on specific events (e.g., "Person" in restricted area).
- **Frustrations**: Laggy video feeds, confusing interfaces, high CPU usage slowing down other monitoring tools.
- **Needs**: High stability, low resource footprint, clear visual indicators (color-coded bounding boxes), performance metrics to confirm the system is running smoothly.

## 2. The AI R&D Developer — "Alex"
- **Background**: Computer Vision engineer prototyping new YOLO models.
- **Goals**: Wants a solid "host" application to test custom-trained YOLO models without rewriting UI code. Needs to observe inference performance (pre-process, inference, post-process timing) to benchmark different models.
- **Frustrations**: Hard-coded model paths, difficult build systems, lack of per-phase timing data, single-threaded architectures that conflate capture and inference bottlenecks.
- **Needs**: Clean code structure, easy model swapping (drop-in ONNX file + `classes.txt`), real-time timing overlay, decoupled capture/inference pipelines for accurate benchmarking.

## 3. The Qt Enthusiast — "Jordan"
- **Background**: C++ developer looking to learn QML and modern C++ best practices.
- **Goals**: Wants to see real-world examples of multi-threaded Qt architecture, Scene Graph rendering, `QAbstractListModel` usage, and `Q_GADGET`/`Q_PROPERTY` patterns.
- **Frustrations**: Examples that do everything in `main.cpp`, block the GUI thread, or use outdated Qt patterns.
- **Needs**: Proper multi-threading examples (worker objects + `moveToThread`), signal/slot cross-thread communication, custom `QQuickItem` with Scene Graph, modern CMake setup with `QML_ELEMENT` registration.
