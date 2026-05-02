# Feature: UI Interaction Logger

**Created**: 2026-05-02 20:35 (UTC+7)

> **Scope**: This document defines the strategy for implementing a lightweight, non-invasive **UI Interaction Logger** that traces every user action and significant application lifecycle event to the debug terminal. The goal is to make it easy to reproduce bugs, understand user flows, and verify that UI actions correctly reach the C++ backend.

---

## 1. Problem Statement

Currently, the only terminal output is from the model lifecycle (`Requesting session creation`, `Session created`). When a bug is reported (e.g., "the task didn't switch"), there is no trace of:
- Which ComboBox was activated and what value was selected.
- Whether the signal reached the C++ controller.
- What the controller did in response.
- How long it took.

This makes it hard to distinguish between:
- A **QML binding bug** (the `onActivated` never fired)
- A **signal-slot wiring bug** (the signal fired but the slot didn't receive it)
- A **business logic bug** (the slot ran but produced the wrong output)

---

## 2. Logging Philosophy

| Principle | Rationale |
|:----------|:----------|
| **Two tiers** | Separate "User Action" logs (UI) from "System Response" logs (C++) so each can be read independently |
| **Structured prefix** | Use a fixed-width tag `[UI]` / `[CTRL]` so you can `grep` either tier |
| **No hot-path logging** | Logs must NEVER appear inside per-frame loops (no `processFrame`, no `readFrame`, no `fpsUpdated`) |
| **QDebug only** | No external library needed. `qDebug()` is zero-overhead in Release builds when disabled via `QT_NO_DEBUG_OUTPUT` |
| **Timestamps** | Prefix with milliseconds since app start so you can measure response latency end-to-end |

### Log Format

```
[HH:MM:SS.mmm] [UI]   <Action>  →  <Value/Target>
[HH:MM:SS.mmm] [CTRL] <Handler> →  <Result>
```

**Example session:**
```
[20:31:05.123] [UI]   ComboBox "Task" activated  →  index=2 ("Seg")
[20:31:05.125] [CTRL] DetectionController::setCurrentTask  →  task=ImageSegmentation
[20:31:05.127] [CTRL] InferenceWorker: Requesting session  →  "assets/openvino/segmentation/yolov8n-seg.xml"
[20:31:07.811] [CTRL] InferenceWorker: Session created     →  OK (2684 ms)
[20:31:08.001] [UI]   ComboBox "Resolution" activated  →  index=1 ("1280x720")
[20:31:08.002] [CTRL] YoloCameraController::setCurrentResolution  →  1280x720
[20:31:12.044] [UI]   Button "Exit" clicked
```

---

## 3. Implementation Strategy

### 3.1 Shared Logger Utility (C++ side)

Create a **header-only** singleton logger to avoid duplicating the timestamp prefix logic everywhere:

```cpp
// shared/domain/UiLogger.h
#pragma once
#include <QDebug>
#include <QDateTime>

namespace UiLogger {
    inline void log(const char* tier, const QString& message) {
        QString timestamp = QDateTime::currentDateTime().toString("HH:mm:ss.zzz");
        qDebug().noquote() << QString("[%1] [%2] %3").arg(timestamp, tier, message);
    }

    inline void ui(const QString& msg)   { log("UI  ", msg); }
    inline void ctrl(const QString& msg) { log("CTRL", msg); }
}
```

Usage:
```cpp
UiLogger::ctrl("DetectionController::setCurrentTask → task=" + QString::number((int)task));
```

---

### 3.2 QML Side: User Action Logs (`[UI]`)

In QML, `console.log()` writes to the debug terminal with the same `qDebug()` backend on Qt 6.

Add a one-liner to each interactive element's handler:

#### Task ComboBox
```qml
StyledComboBox {
    id: taskCombo
    model: ["Detection", "Pose", "Seg"]
    onActivated: (index) => {
        console.log("[UI]   ComboBox 'Task' activated → index=" + index + " (" + model[index] + ")")
        if (!detection) return;
        if (index === 0) detection.currentTask = YoloTask.ObjectDetection
        else if (index === 1) detection.currentTask = YoloTask.PoseEstimation
        else if (index === 2) detection.currentTask = YoloTask.ImageSegmentation
    }
}
```

#### Runtime ComboBox
```qml
StyledComboBox {
    id: runtimeCombo
    model: ["OpenVINO", "ONNX"]
    onActivated: (index) => {
        console.log("[UI]   ComboBox 'Runtime' activated → index=" + index + " (" + model[index] + ")")
        if (!detection) return;
        if (index === 0) detection.currentRuntime = YoloTask.OpenVINO
        else if (index === 1) detection.currentRuntime = YoloTask.ONNXRuntime
    }
}
```

#### Resolution ComboBox
```qml
StyledComboBox {
    id: resCombo
    onActivated: (index) => {
        if (!camera) return;
        var res = camera.supportedResolutions[index];
        console.log("[UI]   ComboBox 'Resolution' activated → " + res.width + "x" + res.height)
        camera.currentResolution = res
    }
}
```

#### Exit Button
```qml
Button {
    text: "Exit Application"
    onClicked: {
        console.log("[UI]   Button 'Exit' clicked")
        Qt.quit()
    }
}
```

---

### 3.3 C++ Side: Controller Response Logs (`[CTRL]`)

#### `DetectionController::setCurrentTask()`
```cpp
void DetectionController::setCurrentTask(YoloTask::TaskType task)
{
    static const QMap<int,QString> taskNames = {
        {1, "ObjectDetection"}, {2, "PoseEstimation"}, {3, "ImageSegmentation"}
    };
    UiLogger::ctrl("DetectionController::setCurrentTask → task=" + taskNames.value((int)task, "Unknown"));

    if (m_currentTask != task) {
        // ... existing code ...
    }
}
```

#### `DetectionController::setCurrentRuntime()`
```cpp
void DetectionController::setCurrentRuntime(YoloTask::RuntimeType runtime)
{
    static const QMap<int,QString> runtimeNames = {{0, "OpenVINO"}, {1, "ONNXRuntime"}};
    UiLogger::ctrl("DetectionController::setCurrentRuntime → " + runtimeNames.value((int)runtime, "Unknown"));
    // ... existing code ...
}
```

#### `YoloCameraController::setCurrentResolution()`
```cpp
void YoloCameraController::setCurrentResolution(const QSize& size)
{
    UiLogger::ctrl("YoloCameraController::setCurrentResolution → " +
                   QString::number(size.width()) + "x" + QString::number(size.height()));
    // ... existing code ...
}
```

#### `InferenceWorker::startInference()` — add session timing
```cpp
void InferenceWorker::startInference(const InferenceConfig& config)
{
    auto t0 = std::chrono::steady_clock::now();

    UiLogger::ctrl("InferenceWorker: Requesting session → \"" + QString::fromStdString(config.modelPath) + "\"");
    const char* status = m_model->createSession(config);

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();

    if (status != nullptr) {
        UiLogger::ctrl("InferenceWorker: Session FAILED → " + QString(status));
        emit errorOccurred("Initialization Error", QString("[YoloPipeline]: %1").arg(status));
        return;
    }

    UiLogger::ctrl("InferenceWorker: Session created → OK (" + QString::number(elapsed) + " ms)");
    m_running = true;
    emit modelLoaded(config.taskType, config.runtimeType);
}
```

---

## 4. Files to Modify

| File | Change |
|:-----|:-------|
| `shared/domain/UiLogger.h` | **NEW** — header-only logger utility |
| `content/Main.qml` | Add `console.log(...)` to each interactive element |
| `features/detection/application/DetectionController.cpp` | Add `UiLogger::ctrl()` in task/runtime setters |
| `features/camera/application/YoloCameraController.cpp` | Add `UiLogger::ctrl()` in resolution setter |
| `features/detection/application/InferenceWorker.cpp` | Replace bare `qDebug()` with `UiLogger::ctrl()` + timing |

---

## 5. Enabling / Disabling in Builds

The `UiLogger` calls compile directly to `qDebug()`. Qt already provides a standard way to silence all debug output in production builds.

### Disable in Release (CMake):
```cmake
# In CMakeLists.txt, for Release builds only:
target_compile_definitions(appCamera PRIVATE
    $<$<CONFIG:Release>:QT_NO_DEBUG_OUTPUT>
)
```

With `QT_NO_DEBUG_OUTPUT` defined, the compiler **eliminates all `qDebug()` calls entirely** — zero binary size cost, zero runtime overhead in Release. No `#ifdef` cluttering source files is needed.

### Verify in Debug build:
```powershell
# Run the debug build to see full log output
.\app\build\Debug\appCamera.bat
```

---

## 6. Expected Output After Implementation

```
[20:31:05.001] [CTRL] InferenceWorker: Requesting session → "assets/openvino/detection/yolov8n.xml"
[20:31:06.880] [CTRL] InferenceWorker: Session created → OK (1879 ms)
[20:31:12.334] [UI  ] ComboBox 'Task' activated → index=2 (Seg)
[20:31:12.336] [CTRL] DetectionController::setCurrentTask → task=ImageSegmentation
[20:31:12.338] [CTRL] InferenceWorker: Requesting session → "assets/openvino/segmentation/yolov8n-seg.xml"
[20:31:14.977] [CTRL] InferenceWorker: Session created → OK (2639 ms)
[20:31:20.501] [UI  ] ComboBox 'Runtime' activated → index=1 (ONNX)
[20:31:20.503] [CTRL] DetectionController::setCurrentRuntime → ONNXRuntime
[20:31:20.505] [CTRL] InferenceWorker: Requesting session → "assets/onnx/segmentation/yolov8n-seg.onnx"
[20:31:21.188] [CTRL] InferenceWorker: Session created → OK (683 ms)
[20:31:30.001] [UI  ] ComboBox 'Resolution' activated → 1280x720
[20:31:30.003] [CTRL] YoloCameraController::setCurrentResolution → 1280x720
[20:31:47.209] [UI  ] Button 'Exit' clicked
```

This single log tells you the **full story** of a session: what the user did, in what order, whether it reached the backend, and how long each model load took.
