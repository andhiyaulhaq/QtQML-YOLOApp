# Feature: Image Input Source

**Created**: 2026-05-03 08:35 (UTC+7)
**Last Modified**: 2026-05-03 08:35 (UTC+7)

> **Scope**: This document defines the implementation strategy for allowing the user to select a **static image file** as the inference input source, integrating it seamlessly into the existing `ICaptureSource` clean architecture pipeline.

---

## 1. Problem Statement

The application currently supports **Live Camera** and **Video File** inputs through the `ICaptureSource` interface. Users also need the ability to load a single static image (e.g., `.jpg`, `.png`) to run YOLO inference on it.

To support image files, we need to:
1. Introduce a new concrete infrastructure implementation (`OpenCVImageFileSource`) that loads an image and serves it to the capture pipeline.
2. Handle the "static" nature of an image: feeding the same frame continuously so the UI and inference pipelines remain active, but doing so efficiently (avoiding 100% CPU usage).
3. Introduce an `ImageFileController` to manage image-specific state, separating it from video and camera concerns.
4. Update the QML UI to provide an image file picker and wire the new controller.

---

## 2. Architecture Strategy

We will follow the same pattern established in the dual-input source refactor. The `CaptureWorker` is already source-agnostic and will seamlessly accept the new image source without any modifications.

### Layer Ownership

```
Presentation (QML):
  Main.qml
    ├── binds to: camera (YoloCameraController)
    ├── binds to: videoFile (VideoFileController)
    └── binds to: imageFile (ImageFileController) ← NEW

Application (Controllers):
  ImageFileController    ← NEW: owns image file path and source activation

Infrastructure:
  OpenCVImageFileSource  ← NEW: loads image via cv::imread and serves it continuously

Domain:
  SourceConfig           ← UPDATE: add ImageFile to InputSourceType enum
```

---

## 3. Domain Changes

### 3.1 Extend `SourceConfig`

```cpp
// features/camera/domain/SourceConfig.h
enum class InputSourceType { 
    LiveCamera, 
    VideoFile, 
    ImageFile   // NEW
};
```

---

## 4. Infrastructure Changes

### 4.1 New: `OpenCVImageFileSource`

**File**: `features/camera/infrastructure/OpenCVImageFileSource.h/.cpp`

Implements `ICaptureSource`. Since an image is static, `readFrame()` will repeatedly return the same loaded `cv::Mat`. 

```cpp
class OpenCVImageFileSource : public ICaptureSource {
public:
    OpenCVImageFileSource() = default;
    ~OpenCVImageFileSource() override;

    bool open(const SourceConfig& config) override;
    void close() override;
    bool readFrame(cv::Mat& outFrame) override;
    QSize currentResolution() const override;

private:
    cv::Mat m_image;
    QString m_filePath;
};
```

**Key implementation notes:**
- `open()`: Uses `cv::imread(config.filePath.toStdString(), cv::IMREAD_COLOR)` to load the image into memory once.
- `readFrame()`: Clones or copies `m_image` to `outFrame`.
- **CPU Pacing**: Since returning a cached image is practically instantaneous, the `CaptureWorker` might run a tight loop and consume excessive CPU. To mitigate this, `OpenCVImageFileSource` could artificially pace itself (e.g., simulating 30 FPS by sleeping, or the worker could pace it). However, since `CaptureWorker` already relies on the source to pace (or paces video files natively), we should introduce a short sleep (e.g., `QThread::msleep(33)`) inside `readFrame()` for static images to simulate 30 FPS.

---

## 5. Application Changes

### 5.1 New: `ImageFileController`

**File**: `features/camera/application/ImageFileController.h/.cpp`

Identical in structure to `VideoFileController`.

```cpp
class ImageFileController : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString filePath READ filePath NOTIFY filePathChanged)
    Q_PROPERTY(bool hasFile READ hasFile NOTIFY filePathChanged)

public:
    explicit ImageFileController(QObject *parent = nullptr);

    Q_INVOKABLE void setFilePath(const QUrl& fileUrl);
    Q_INVOKABLE void activate();

signals:
    void filePathChanged();
    void sourceReadyRequested(ICaptureSource* source, SourceConfig config);

private:
    QString m_filePath;
    std::unique_ptr<OpenCVImageFileSource> m_source;
};
```

---

## 6. Presentation Changes

### 6.1 QML Bindings & File Dialog

1. **ComboBox Update**: Add "Image File" to the model.
2. **File Dialog**: Add an `ImageDialog` (or reuse `FileDialog` with different filters) for `.jpg, .png, .jpeg`.
3. **UI Layout**: Mirror the `Video File` behavior. If `Image File` is selected, show the file path and a "Browse..." button. Disable the resolution picker.

```qml
// Main.qml
FileDialog {
    id: imageFileDialog
    title: "Select Image File"
    nameFilters: ["Image files (*.jpg *.jpeg *.png)"]
    onAccepted: {
        inputMode = "image"
        imageFile.setFilePath(selectedFile)
    }
}
```

---

## 7. `AppController` Wiring

```cpp
// AppController.cpp
void AppController::setupCamera() {
    // ... existing setup
    m_imageFileController = new ImageFileController(this);
    m_engine->rootContext()->setContextProperty("imageFile", m_imageFileController);
}

void AppController::wireEverything() {
    // ... existing wiring
    connect(m_imageFileController, &ImageFileController::sourceReadyRequested,
            m_captureWorker, &CaptureWorker::setSource);
}
```

---

## 8. Implementation Plan (Task Order)

| Step | Task | Files |
|:-----|:-----|:------|
| 1 | Add `ImageFile` to `InputSourceType` enum | `domain/SourceConfig.h` |
| 2 | Create `OpenCVImageFileSource` (implement `ICaptureSource` with 30fps pacing) | `infrastructure/OpenCVImageFileSource.h/.cpp` |
| 3 | Create `ImageFileController` | `application/ImageFileController.h/.cpp` |
| 4 | Register and wire `ImageFileController` in `AppController` | `shared/application/AppController.h/.cpp` |
| 5 | Update `Main.qml`: Add Image mode to ComboBox, add `imageFileDialog`, update UI bindings | `content/Main.qml` |
| 6 | Register new source files in `CMakeLists.txt` | `CMakeLists.txt` |

---

## 9. Edge Cases & Mitigations

| Case | Mitigation |
|:-----|:-----------|
| **100% CPU Usage** | `CaptureWorker` or `OpenCVImageFileSource::readFrame()` must enforce pacing (e.g., `QThread::msleep(33)`) since reading from RAM is instantaneous. |
| **Unsupported Image Format** | `cv::imread` returns an empty matrix. `open()` should return `false` and log the error. |
| **Image Resolution Mismatch** | YOLO pipeline and UI overlay already handle normalized coordinates; the image will be aspect-fit in the UI automatically. |
