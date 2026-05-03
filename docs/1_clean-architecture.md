# Clean Architecture — YOLOApp

**Last Modified**: 2026-05-03 22:35 (UTC+7)

> **Scope**: This document defines the canonical clean architecture for the **QtOpenCVCamera / YOLOApp** C++/Qt/QML desktop application. It is the single source of truth for structural decisions, layer boundaries, dependency rules, and feature organization.

---

## 1. Philosophy & Goals

| Principle | Rationale |
|:----------|:----------|
| **Dependency inversion** | Inner layers (domain) never depend on outer layers (infrastructure, UI). Interfaces always point inward. |
| **Separation of concerns** | AI pipeline, camera hardware, UI rendering, and system metrics are completely isolated from each other. |
| **Testability** | Business logic lives in pure C++ classes with no Qt/OpenCV coupling, enabling unit tests without a running application. |
| **Feature cohesion** | Code is organized by **feature** (inference, capture, monitoring), not by technical type (controllers, models, views). Each feature folder is a self-contained vertical slice. |
| **Performance preservation** | The architecture adds no runtime overhead. Interfaces and strategy patterns are resolved at compile/link time wherever possible. |

---

## 2. Layer Model

The application is organized into four concentric layers. Dependencies flow strictly **inward** (outer → inner). No inner layer may import from an outer layer.

```
┌────────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                         │
│          QML Files · QQuickItem · Q_PROPERTY bindings          │
│             (features/*/ui/*.qml, features/*/ui/*.h)           │
├────────────────────────────────────────────────────────────────┤
│                    APPLICATION LAYER                           │
│     Qt controllers (QObject) · Worker threads · Mediators      │
│      (features/*/application/*.h, features/*/application/*.cpp)│
├────────────────────────────────────────────────────────────────┤
│                      DOMAIN LAYER                              │
│   Pure C++ interfaces, data structs, business rules / models   │
│        (features/*/domain/*.h, shared/domain/*.h)              │
├────────────────────────────────────────────────────────────────┤
│                  INFRASTRUCTURE LAYER                          │
│  OpenCV · ONNX Runtime · OpenVINO · PDH/PSAPI · QVideoSink    │
│   (features/*/infrastructure/*.h, features/*/infrastructure/*.cpp) │
└────────────────────────────────────────────────────────────────┘
```

### Layer Contracts

| Layer | May import from | Must NOT import from |
|:------|:----------------|:---------------------|
| **Presentation** | Application, Domain | Infrastructure directly |
| **Application** | Domain, Infrastructure (via interface) | Presentation |
| **Domain** | Nothing external | Application, Infrastructure, Presentation |
| **Infrastructure** | Domain (interfaces/types only) | Application, Presentation |

---

## 3. Feature-First Project Structure

Each top-level folder under `src/features/` is a self-contained **feature module** that owns its own domain model, application logic, infrastructure adapter, and UI bridge.

```
app/
├── CMakeLists.txt
├── src/
├── main.cpp                        # Entry point — wires QML engine only
│   │
│   ├── features/                       # Feature modules (vertical slices)
│   │   │
│   │   ├── inference/                  # ── INFERENCE FEATURE ──────────────
│   │   │   ├── domain/
│   │   │   │   ├── VisionObject.h      # Q_GADGET value type (normalized coords)
│   │   │   │   ├── InferenceResult.h   # Pure-C++ raw inference result
│   │   │   │   ├── InferenceConfig.h   # Init params
│   │   │   │   ├── InferenceTiming.h   # Per-phase timing struct
│   │   │   │   ├── TaskType.h          # Enum: Detect / Pose / Segment
│   │   │   │   └── IInferenceModel.h   # Interface: runInference(frame) → results
│   │   │   │
│   │   │   ├── application/
│   │   │   │   ├── InferenceWorker.h   # QObject worker, lives on inference thread
│   │   │   │   ├── InferenceWorker.cpp
│   │   │   │   ├── InferenceController.h  # QML_ELEMENT, exposes detections/timing/runtime
│   │   │   │   └── InferenceController.cpp
│   │   │   │
│   │   │   ├── infrastructure/
│   │   │   │   ├── IInferenceBackend.h    # Strategy interface (pure C++)
│   │   │   │   ├── OnnxRuntimeBackend.h
│   │   │   │   ├── OnnxRuntimeBackend.cpp
│   │   │   │   ├── OpenVinoBackend.h
│   │   │   │   ├── OpenVinoBackend.cpp
│   │   │   │   ├── YoloPipeline.h         # Facade: preprocess → infer → postprocess
│   │   │   │   ├── YoloPipeline.cpp
│   │   │   │   ├── PreProcessor.h
│   │   │   │   ├── PreProcessor.cpp
│   │   │   │   ├── PostProcessor.h        # IPostProcessor + concrete strategies
│   │   │   │   ├── PostProcessor.cpp
│   │   │   │   └── SimdUtils.h            # SSE4.1 intrinsics (header-only)
│   │   │   │
│   │   │   └── ui/
│   │   │       ├── InferenceListModel.h   # QAbstractListModel bridge
│   │   │       ├── InferenceListModel.cpp
│   │   │       ├── InferenceOverlayItem.h # QQuickItem Scene Graph renderer
│   │   │       └── InferenceOverlayItem.cpp
│   │   │
│   │   ├── capture/                    # ── CAPTURE FEATURE ────────────────
│   │   │   ├── domain/
│   │   │   │   ├── SourceConfig.h      # Input type, resolution, file path
│   │   │   │   ├── CameraFrame.h       # Value type wrapping shared_ptr<cv::Mat>
│   │   │   │   └── ICaptureSource.h    # Interface: open/close/readFrame()
│   │   │   │
│   │   │   ├── application/
│   │   │   │   ├── CaptureWorker.h     # QObject worker, lives on capture thread
│   │   │   │   ├── CaptureWorker.cpp
│   │   │   │   ├── YoloCameraController.h  # QML_ELEMENT: fps, resolution, videoSink
│   │   │   │   ├── YoloCameraController.cpp
│   │   │   │   ├── VideoFileController.h
│   │   │   │   └── VideoFileController.cpp
│   │   │   │
│   │   │   └── infrastructure/
│   │   │       ├── OpenCVCameraSource.h   # ICaptureSource → cv::VideoCapture
│   │   │       ├── OpenCVCameraSource.cpp
│   │   │       ├── OpenCVImageFileSource.h # ICaptureSource → cv::imread
│   │   │       ├── OpenCVImageFileSource.cpp
│   │   │       ├── FFmpegVideoSource.h    # ICaptureSource → libavcodec/format
│   │   │       └── FFmpegVideoSource.cpp
│   │   │
│   │   └── monitoring/                 # ── MONITORING FEATURE ──────────────
│   │       ├── domain/
│   │       │   ├── SystemStats.h       # Value type: cpuPercent, sysMem, procMem
│   │       │   └── ISystemMonitor.h    # Interface: startMonitoring / stopMonitoring
│   │       │
│   │       ├── application/
│   │       │   ├── SystemMonitorWorker.h    # QObject worker, lives on monitor thread
│   │       │   └── SystemMonitorWorker.cpp
│   │       │
│   │       └── infrastructure/
│   │           ├── WindowsSystemMonitor.h   # PDH + PSAPI implementation
│   │           ├── WindowsSystemMonitor.cpp
│   │           ├── LinuxSystemMonitor.h     # /proc implementation
│   │           └── LinuxSystemMonitor.cpp
│   │
│   └── shared/                         # Cross-cutting, no feature affiliation
│       ├── domain/
│       │   └── AppConfig.h             # Compile-time constants (frame/model dims)
│       └── application/
│           ├── AppController.h         # Root QML_ELEMENT orchestrating features
│           └── AppController.cpp
│
└── src/ui/                             # QML UI files
    └── Main.qml                        # Root window, flat structure for now

---

## 4. Layer Detail: Domain

The Domain layer contains **no framework dependencies**. Every class is a plain C++ struct or pure-virtual interface.

### 4.1 Inference Domain

```cpp
// Raw inference result in pixel-space
struct InferenceResult {
    int        classId;
    float      confidence;
    cv::Rect   box;
    std::vector<cv::Point2f> keyPoints;  // pose
    cv::Mat    boxMask;                  // segmentation
};

// Internal struct for coordinate mapping
struct LetterboxInfo {
    float scale = 1.0f;
    int padW = 0;
    int padH = 0;
};

// features/inference/domain/VisionObject.h
// Normalized [0,1] coordinate value-type for QML (Q_GADGET)
struct VisionObject {
    Q_GADGET
    Q_PROPERTY(int classId ...)
    Q_PROPERTY(float confidence ...)
    Q_PROPERTY(QString label ...)
    Q_PROPERTY(float x ...)
    Q_PROPERTY(float y ...)
    Q_PROPERTY(float w ...)
    Q_PROPERTY(float h ...)
    Q_PROPERTY(QList<QPointF> keyPoints ...)
};

// features/inference/domain/IInferenceModel.h
// Pure interface — infrastructure implements, application consumes
class IInferenceModel {
public:
    virtual ~IInferenceModel() = default;
    virtual const char* createSession(const InferenceConfig& config) = 0;
    virtual char* runInference(const cv::Mat& frame,
                               std::vector<InferenceResult>& results,
                               InferenceTiming& timing) = 0;
    virtual const std::vector<std::string>& classNames() const = 0;
    virtual void warmUp() = 0;
};
```

### 4.2 Capture Domain

```cpp
// features/capture/domain/ICaptureSource.h
class ICaptureSource {
public:
    virtual ~ICaptureSource() = default;
    virtual bool open(const SourceConfig& config) = 0;
    virtual void close() = 0;
    virtual bool readFrame(cv::Mat& outFrame) = 0;
    virtual QSize currentResolution() const = 0;
    virtual int64_t frameCount() const;
    virtual double nativeFps() const;
};
```

### 4.3 Monitoring Domain

```cpp
// features/monitoring/domain/ISystemMonitor.h
class ISystemMonitor {
public:
    virtual ~ISystemMonitor() = default;
    virtual void initialize() = 0;
    virtual void cleanup() = 0;
    virtual SystemStats poll() = 0;
};
```

---

## 5. Layer Detail: Infrastructure

Infrastructure classes implement domain interfaces using concrete third-party libraries. They **inject dependencies** (via constructor or factory) into Application layer workers.

### 5.1 Inference Infrastructure

```
IInferenceModel  ←  YoloPipeline
                          │
               ┌──────────┴──────────┐
               │                     │
    IInferenceBackend       IPostProcessor
              │                     │
    ┌─────────┴──────┐    ┌─────────┴──────────────────┐
    │                │    │          │                   │
    │                │    │          │                   │
 OnnxRuntime   OpenVINO  Detect   Pose           Segmentation
 Backend       Backend   PostProcessor PostProcessor PostProcessor
```

### 5.2 Capture Infrastructure

`OpenCVCameraSource` and `FFmpegVideoSource` implement `ICaptureSource`. They own the frame ring buffers and multi-buffered `QVideoFrame` allocation (4-frame pool), keeping all OpenCV/FFmpeg coupling inside the infrastructure layer.

### 5.3 Monitoring Infrastructure

Platform-specific implementations are selected at compile time via `#ifdef` or CMake target selection — the `ISystemMonitor` interface is always the same regardless of host OS.

---

## 6. Layer Detail: Application

Application layer classes are the only ones allowed to own `QObject`, `QThread`, and Qt cross-thread plumbing. Workers coordinate infrastructure + domain; Controllers expose state to QML.

### 6.1 Threading Model

```
┌──────────────────────────────────────────────────────────────────┐
│  Main Thread (Qt Event Loop / GUI)                                │
│  AppController · YoloCameraController · InferenceController       │
│  InferenceListModel · InferenceOverlayItem                        │
│  Constraint: never block > 16 ms                                  │
└────────┬───────────────────┬──────────────────┬──────────┘
         │ QueuedConnection  │ QueuedConnection  │ QueuedConnection
    ┌────▼────────┐   ┌──────▼──────┐   ┌───────▼──────┐
    │CaptureWorker│   │InferenceWork│   │SystemMonitor │
    │ (Normal P.) │   │er (High P.) │   │Worker (Low P)│
    │             │   │             │   │              │
    │ OpenCV      │   │ YoloPipeline│   │ PDH/PSAPI    │
    │ VideoCapture│   │ OnnxRuntime │   │ /proc        │
    └─────────────┘   └─────────────┘   └──────────────┘
```

### 6.2 AppController

`AppController` is the single root `QML_ELEMENT`. It:
- Holds and owns `YoloCameraController` and `InferenceController` as properties.
- Wires the `CaptureWorker::frameReady` signal to `InferenceWorker::processFrame`.
- Wires `InferenceWorker::detectionsReady` to `InferenceController::updateDetections`.
- Manages shared `QThread` lifetimes.

### 6.3 YoloCameraController

Exposes to QML:
- `videoSink` (write-once, triggers worker startup)
- `cameraFps` (double, updated from CaptureWorker)
- `currentResolution` (QSize, R/W)
- `supportedResolutions` (QVariantList)

### 6.4 InferenceController

Exposes to QML:
- `detections` (QObject* → InferenceListModel)
- `currentTask` (TaskType enum)
- `currentRuntime` (RuntimeType enum)
- `preProcessTime`, `inferenceTime`, `postProcessTime` (double, ms)
- `inferenceFps` (double)

---

## 7. Layer Detail: Presentation

The QML layer reads Controller properties and never contains logic. All computations happen in C++.

### 7.1 QML Composition Strategy

```qml
// Main.qml
Window {
    AppController { id: app }

    // Flat composition in Main.qml for now
    VideoOutput { id: videoOutput ... }
    InferenceOverlayItem { inferenceController: inference ... }
    
    // HUDs
    Text { text: "FPS: " + camera.cameraFps ... }
}
```

---

## 8. Dependency Graph (Cross-Feature)

```mermaid
graph TD
    subgraph Presentation
        Main["Main.qml"]
        CamView["CameraView.qml"]
        DetOverlay["YoloOverlay.qml"]
    end

    subgraph Application
        AppCtrl["AppController"]
        CamCtrl["YoloCameraController"]
        DetCtrl["InferenceController"]
        CaptureW["CaptureWorker"]
        InferW["InferenceWorker"]
        MonitorW["SystemMonitorWorker"]
    end

    subgraph Domain
        ICapSrc["ICaptureSource"]
        IInfModel["IInferenceModel"]
        ISysMon["ISystemMonitor"]
        VisionObj["VisionObject (Q_GADGET)"]
    end

    subgraph Infrastructure
        CaptureSrc["CaptureSource Implementation"]
        YoloPipe["YoloPipeline"]
        WinMon["WindowsSystemMonitor"]
    end

    Main --> AppCtrl
    Main --> CamView
    Main --> DetOverlay

    AppCtrl --> CamCtrl
    AppCtrl --> DetCtrl

    CamCtrl --> CaptureW
    DetCtrl --> InferW
    AppCtrl --> MonitorW

    CaptureW --> ICapSrc
    InferW --> IInfModel
    MonitorW --> ISysMon

    ICapSrc --> CaptureSrc
    IInfModel --> YoloPipe
    ISysMon --> WinMon

    DetCtrl --> VisionObj
    DetOverlay --> DetCtrl
    CamView --> CamCtrl
```

---

## 9. Inter-Feature Communication

Features communicate exclusively through the **Application layer**. No feature's domain or infrastructure may import from another feature's domain or infrastructure.

```
camera::CaptureWorker  ──frameReady()──►  inference::InferenceWorker
                                               (wired by AppController)

inference::InferenceWorker  ──detectionsReady()──►  inference::InferenceController
                                                     ──latestDetectionsReady()──►  camera::CaptureWorker
                                                              (for on-frame overlay blending)
```

---

## 10. Design Patterns Inventory

| Pattern | Location | Purpose |
|:--------|:---------|:--------|
| **Strategy** | `IInferenceBackend` → `OnnxRuntimeBackend` / `OpenVinoBackend` | Swap inference runtime without touching pipeline |
| **Strategy** | `IPostProcessor` → `DetectionPostProcessor` / `PosePostProcessor` / `SegmentationPostProcessor` | Task-specific post-processing |
| **Facade** | `YoloPipeline` | Single entry point for pre→infer→post pipeline |
| **Template Method** | `IPostProcessor::PostProcess` | Skeleton algorithm with task-specific overrides |
| **Factory** | `YoloPipeline::CreateSession` | Selects backend + post-processor based on `InferenceConfig` |
| **Observer / Reactive** | Qt signals/slots across threads | Decoupled event propagation without shared state |
| **Repository / Model** | `InferenceListModel` | Normalizes and serves inference data to QML |
| **Worker Thread** | `CaptureWorker`, `InferenceWorker`, `SystemMonitorWorker` | Offload blocking ops from GUI thread |
| **Adapter** | `OpenCVCameraSource` | Adapts `cv::VideoCapture` to `ICaptureSource` |
| **Value Object** | `VisionObject`, `SystemStats`, `CameraFrame` | Immutable data carriers with no behavior |

---

## 11. Key Architectural Rules (Enforced)

1. **No `#include` of Qt headers inside `domain/`** — domain is framework-agnostic.
2. **No `cv::Mat` or ONNX types inside `application/`** — only domain types cross this boundary.
3. **No business logic in QML** — controllers expose computed properties only.
4. **Workers never access `QGuiApplication`** — all GUI interactions go through signals to the main thread.
5. **`AppConfig.h` is the only global constant header** — no magic numbers scattered in code.
6. **One `QML_ELEMENT` per Controller** — each feature exposes exactly one root QML element.
7. **`shared/` has no feature dependencies** — it is a true leaf with zero upstream imports.

---

## 12. File → Responsibility Map

| File | Layer | Responsibility |
|:-----|:------|:---------------|
| `features/inference/domain/IInferenceModel.h` | Domain | Contract for any YOLO-compatible model |
| `features/inference/domain/VisionObject.h` | Domain | Normalized, QML-safe inference value object |
| `features/inference/infrastructure/YoloPipeline.h` | Infrastructure | Facade orchestrating ONNX/OpenVINO inference |
| `features/inference/infrastructure/OnnxRuntimeBackend.h` | Infrastructure | ONNX Runtime session pool + inference |
| `features/inference/infrastructure/OpenVinoBackend.h` | Infrastructure | OpenVINO compiled model + infer request |
| `features/inference/application/InferenceWorker.h` | Application | Thread worker: receives frames → runs pipeline |
| `features/inference/application/InferenceController.h` | Application | QML_ELEMENT: exposes task/runtime/timing/detections |
| `features/inference/ui/InferenceListModel.h` | Presentation | QAbstractListModel bridging results to QML |
| `features/inference/ui/InferenceOverlayItem.h` | Presentation | QQuickItem scene-graph renderer |
| `features/capture/domain/ICaptureSource.h` | Domain | Contract for any camera or video adapter |
| `features/capture/infrastructure/OpenCVCameraSource.h` | Infrastructure | OpenCV VideoCapture (DirectShow) adapter |
| `features/capture/infrastructure/FFmpegVideoSource.h` | Infrastructure | Native FFmpeg (libav) high-performance adapter |
| `features/capture/application/CaptureWorker.h` | Application | Thread worker: captures frames → feeds inference |
| `features/capture/application/YoloCameraController.h` | Application | QML_ELEMENT: exposes fps/resolution/videoSink |
| `features/monitoring/domain/ISystemMonitor.h` | Domain | Contract for platform resource polling |
| `features/monitoring/infrastructure/WindowsSystemMonitor.h` | Infrastructure | PDH CPU + PSAPI memory |
| `features/monitoring/application/SystemMonitorWorker.h` | Application | Timer-driven worker emitting SystemStats |
| `features/monitoring/application/MonitoringController.h` | Application | QML_ELEMENT: exposes statsText to UI |
| `shared/application/AppController.h` | Application | Root orchestrator: wires all feature controllers |

| `shared/domain/AppConfig.h` | Domain | Compile-time constants (frame size, model dims) |
| `src/ui/Main.qml` | Presentation | Root QML window |

---

## 13. Glossary

| Term | Definition |
|:-----|:-----------|
| **Feature module** | A vertical slice of functionality owning domain, application, infrastructure, and UI sub-layers |
| **Domain** | Pure C++ layer: interfaces, value objects, business rules; no framework dependencies |
| **Application** | Qt layer: QObject workers, controllers, thread management; orchestrates domain + infrastructure |
| **Infrastructure** | Third-party adapters: OpenCV, ONNX Runtime, OpenVINO, PDH/PSAPI |
| **Presentation** | QML files + QQuickItem/QAbstractListModel bridges; reads controller properties only |
| **Controller** | A `QML_ELEMENT` QObject exposing feature state as Qt properties to the QML layer |
| **Worker** | A `QObject` designed to live on a background `QThread`; performs blocking I/O or computation |
| **Strategy** | A polymorphic algorithm family selectable at runtime via a shared interface |
| **Facade** | A simplified front-end to a complex subsystem (e.g., `YoloPipeline`) |
