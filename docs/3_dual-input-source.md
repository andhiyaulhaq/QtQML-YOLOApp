# Feature: Dual Input Source — Live Camera & Video File

**Created**: 2026-05-02 20:30 (UTC+7)
**Last Modified**: 2026-05-02 22:16 (UTC+7)

> **Scope**: This document defines the complete implementation strategy for allowing the user to switch between a **live camera feed** and a **local video file** as the inference input source, while preserving all existing architecture constraints and zero regressions.

---

## 1. Problem Statement

The current `ICameraSource` contract is designed around a hardware camera (`deviceId`). The `CameraConfig` struct only carries `deviceId`, `resolution`, and `fps`. There is no concept of a file path as an input source.

To support video files, we need to:
1. Introduce a new concrete infrastructure implementation of `ICaptureSource` that reads from a file.
2. Allow the user to pick a file via a native OS dialog from QML.
3. Allow the `CaptureWorker` to swap the active source at runtime without restarting threads.
4. Handle video-specific concerns: **end-of-file looping**, **native FPS pacing**, and **seek-to-start on repeat**.

> **Architecture Decision (v2)**: A dedicated `VideoFileController` is introduced for video file management, completely separate from `YoloCameraController`. This adheres to the Single Responsibility Principle — `YoloCameraController` owns hardware camera concerns (device enumeration, resolution switching), while `VideoFileController` owns file-based playback concerns (file path, looping state). The `CaptureWorker` is source-agnostic and receives a new `ICaptureSource*` from whichever controller is active.

---

## 2. Architecture Strategy

The core principle: **respect the existing `ICameraSource` abstraction**. The `CaptureWorker` already depends on the interface, not the concrete class.

### Layer Ownership

```
Presentation (QML):
  Main.qml
    ├── binds to: camera (YoloCameraController)   → live camera controls
    ├── binds to: videoFile (VideoFileController) → file path, open dialog
    └── binds to: capture (shared FPS/mode state)

Application (Controllers):
  YoloCameraController   ← owns: resolution picker, camera FPS, device switching
  VideoFileController    ← owns: filePath, looping, video FPS pacing

Infrastructure:
  OpenCVCameraSource     ← live webcam via OpenCV
  OpenCVVideoFileSource  ← video file via OpenCV

Domain:
  ICaptureSource         ← pure interface (Renamed from ICameraSource)
  SourceConfig           ← data struct (Renamed from CameraConfig)

Shared:
  CaptureWorker          ← source-agnostic; receives setSource(ICaptureSource*)
  AppController          ← composition root; wires all signals
```

### High-Level Flow: Camera → Video File Switch

```
User selects "Video" in Source ComboBox
        │
        ▼
QML opens FileDialog → onAccepted → videoFile.setFilePath(url)
        │
        ▼
VideoFileController converts QUrl → local path string
        │  emits: sourceReadyRequested(OpenCVVideoFileSource*)
        ▼
AppController wires → CaptureWorker::setSource(ICameraSource*)
        │  atomic source swap under m_sourceMutex
        ▼
CaptureWorker continues hot loop with new source (no thread restart)
```

### High-Level Flow: Video File → Camera Switch

```
User selects "Camera" in Source ComboBox
        │
        ▼
QML calls camera.activate()
        │
        ▼
YoloCameraController emits: sourceReadyRequested(OpenCVCameraSource*)
        │
        ▼
AppController wires → CaptureWorker::setSource(ICameraSource*)
```

---

## 3. Domain Changes

### 3.1 Rename and Extend `SourceConfig`

```cpp
// features/camera/domain/SourceConfig.h (Renamed from CameraConfig.h)
enum class InputSourceType { LiveCamera, VideoFile };

struct SourceConfig {
    InputSourceType sourceType = InputSourceType::LiveCamera;
    int             deviceId   = 0;
    QString         filePath   = "";       // used when sourceType == VideoFile
    QSize           resolution = QSize(640, 480);
    double          fps        = 30.0;
    bool            loop       = true;     // replay video when EOF reached
};
```

> **Why in domain?** `SourceConfig` is a pure data struct with no framework coupling. Adding `InputSourceType` and `filePath` keeps it framework-agnostic.

---

## 4. Infrastructure Changes

### 4.1 New: `OpenCVVideoFileSource`

**File**: `features/camera/infrastructure/OpenCVVideoFileSource.h/.cpp`

Implements `ICaptureSource`, same contract as `OpenCVCameraSource`.

```cpp
class OpenCVVideoFileSource : public ICaptureSource {
public:
    OpenCVVideoFileSource() = default;
    ~OpenCVVideoFileSource() override;

    bool open(const SourceConfig& config) override;   // uses config.filePath
    void close() override;
    bool readFrame(cv::Mat& outFrame) override;
    QSize currentResolution() const override;

    double nativeFps() const;   // returns the file's encoded FPS
};
```

**Key implementation notes:**
- `open()` calls `m_capture.open(config.filePath.toStdString())`.
- `readFrame()`: if `m_capture.read()` returns false and `config.loop` is true, seek to frame 0 with `m_capture.set(cv::CAP_PROP_POS_FRAMES, 0)` and retry.
- `nativeFps()` reads `cv::CAP_PROP_FPS` after opening; falls back to 30.0 if zero.

### 4.2 Existing: `OpenCVCameraSource` (unchanged)

No modifications needed to the camera source infrastructure.

---

## 5. Application Changes

### 5.1 `CaptureWorker` — Source-Agnostic Swap Interface

`CaptureWorker` drops all knowledge of "camera vs. video" mode. It exposes a single clean swap method:

```cpp
// CaptureWorker.h — simplified API
public slots:
    void setSource(ICaptureSource* source, const SourceConfig& config);
    void stopCapturing();
```

**Implementation strategy:**
1. Acquire `m_sourceMutex`.
2. Close the previous source.
3. Set `m_source` to the new pointer (ownership stays with the controller that created it).
4. Store the new config; set `m_sourceUpdatePending = true`.
5. The capture loop detects the flag, calls `openCamera(config)`, and continues.

**FPS pacing (video file mode only):**
- In the capture loop, check `config.sourceType == InputSourceType::VideoFile`.
- Read `nativeFps()` from the source; sleep `max(0, targetMs - elapsed)` per frame to prevent running faster than real-time.

### 5.2 `YoloCameraController` — Camera-Only Controller

Responsibilities **strictly limited to**:
- `supportedResolutions()` — hardware resolution list
- `currentResolution` / `setCurrentResolution()` — camera hardware config
- `cameraFps` — live FPS from `CaptureWorker`
- `activate()` — creates `OpenCVCameraSource`, emits `sourceReadyRequested`

**Remove** from `YoloCameraController`:
- `inputSourceMode` property
- `videoFilePath` property
- `setVideoFilePath()` slot
- `switchToVideoFile()` / `switchToLiveCamera()` slots

### 5.3 New: `VideoFileController` — File Playback Controller

**File**: `features/camera/application/VideoFileController.h/.cpp`

```cpp
class VideoFileController : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString filePath READ filePath NOTIFY filePathChanged)
    Q_PROPERTY(bool hasFile READ hasFile NOTIFY filePathChanged)

public:
    Q_INVOKABLE void setFilePath(const QUrl& fileUrl);

signals:
    void filePathChanged();
    // Emitted when a valid file is ready — AppController wires this to CaptureWorker
    void sourceReadyRequested(ICaptureSource* source, SourceConfig config);

private:
    QString m_filePath;
    std::unique_ptr<OpenCVVideoFileSource> m_source;

    bool hasFile() const { return !m_filePath.isEmpty(); }
    QString filePath() const { return m_filePath; }
};
```

`setFilePath()` validates the path, constructs an `OpenCVVideoFileSource`, and emits `sourceReadyRequested`.

### 5.4 `YoloCameraController` — `activate()` Signal

```cpp
// YoloCameraController.h
signals:
    void sourceReadyRequested(ICaptureSource* source, SourceConfig config);

public slots:
    void activate();   // creates OpenCVCameraSource, emits sourceReadyRequested
```

---

## 6. Presentation Changes

### 6.1 QML Bindings

```qml
// Main.qml — expose both controllers
required property YoloCameraController camera
required property VideoFileController videoFile

// Source ComboBox
StyledComboBox {
    model: ["Camera", "Video"]
    onActivated: (index) => {
        if (index === 0) camera.activate()
        else fileDialog.open()
    }
}

// File picker
FileDialog {
    id: fileDialog
    nameFilters: ["Video files (*.mp4 *.avi *.mkv *.mov *.wmv)"]
    onAccepted: videoFile.setFilePath(selectedFile)
}
```

### 6.2 Resolution Picker Behavior

- **Camera mode**: resolution picker is **enabled**; driven by `camera.supportedResolutions`.
- **Video file mode**: resolution picker is **disabled**; video has its own fixed resolution.
- Mode is tracked by a simple `string` property on a shared state object or derived from which controller was last activated.

---

## 7. `AppController` Wiring

```cpp
void AppController::wireEverything() {
    // Camera source switching
    connect(m_cameraController, &YoloCameraController::sourceReadyRequested,
            m_captureWorker, &CaptureWorker::setSource);

    // Video file source switching
    connect(m_videoFileController, &VideoFileController::sourceReadyRequested,
            m_captureWorker, &CaptureWorker::setSource);

    // FPS feedback (both modes)
    connect(m_captureWorker, &CaptureWorker::fpsUpdated,
            m_cameraController, &YoloCameraController::updateFps);

    // Resolution feedback (camera mode only)
    connect(m_captureWorker, &CaptureWorker::resolutionChanged,
            m_cameraController, &YoloCameraController::handleResolutionChanged);
}
```

---

## 8. Implementation Plan (Task Order)

| Step | Task | Files |
|:-----|:-----|:------|
| 1 | Rename and Extend `SourceConfig` with `InputSourceType` and `filePath` | `domain/SourceConfig.h` |
| 2 | Rename `ICameraSource` to `ICaptureSource` | `domain/ICaptureSource.h` |
| 3 | Create `OpenCVVideoFileSource` | `infrastructure/OpenCVVideoFileSource.h/.cpp` |
| 4 | Refactor `CaptureWorker`: replace switch slots with `setSource(ICaptureSource*, SourceConfig)` | `application/CaptureWorker.h/.cpp` |
| 5 | Add FPS pacing throttle in capture loop for video file mode | `application/CaptureWorker.cpp` |
| 6 | Strip video-related code from `YoloCameraController`; add `activate()` | `application/YoloCameraController.h/.cpp` |
| 7 | Create `VideoFileController` | `application/VideoFileController.h/.cpp` |
| 8 | Register both controllers in `AppController`; wire signals | `shared/application/AppController.cpp` |
| 9 | Update `Main.qml`: source ComboBox, FileDialog, disable resolution in video mode | `content/Main.qml` |
| 10 | Register `Qt6::QuickDialogs2` + new source files in `CMakeLists.txt` | `CMakeLists.txt` |

---

## 9. Edge Cases & Mitigations

| Case | Mitigation |
|:-----|:-----------|
| User selects a corrupt/unreadable file | `OpenCVVideoFileSource::open()` returns `false`; `VideoFileController` does not emit `sourceReadyRequested`; shows an error state |
| Video has no audio track | Not relevant — only video frames are read |
| Video FPS is 0 (some MJPEG files) | Default to 30 FPS fallback in `nativeFps()` |
| User switches source while inference is mid-frame | `m_sourceMutex` in `CaptureWorker` prevents torn reads |
| Video file resolution differs from camera resolution | Overlay uses **normalized** coordinates — adapts automatically |
| Resolution picker shows stale values in video mode | Disable picker when video source is active |

---

## 10. Files to Create / Modify

| File | Action |
|:-----|:-------|
| `features/camera/domain/SourceConfig.h` | **RENAME** — from `CameraConfig.h`; add `InputSourceType` enum + `filePath`, `loop` fields |
| `features/camera/domain/ICaptureSource.h` | **RENAME** — from `ICameraSource.h` |
| `features/camera/infrastructure/OpenCVVideoFileSource.h` | **NEW** |
| `features/camera/infrastructure/OpenCVVideoFileSource.cpp` | **NEW** |
| `features/camera/application/CaptureWorker.h` | **MODIFY** — replace switch slots with `setSource()` |
| `features/camera/application/CaptureWorker.cpp` | **MODIFY** — source swap logic, FPS throttle |
| `features/camera/application/YoloCameraController.h` | **MODIFY** — strip video state, add `activate()` |
| `features/camera/application/YoloCameraController.cpp` | **MODIFY** — implement `activate()` |
| `features/camera/application/VideoFileController.h` | **NEW** |
| `features/camera/application/VideoFileController.cpp` | **NEW** |
| `shared/application/AppController.cpp` | **MODIFY** — wire new controllers |
| `content/Main.qml` | **MODIFY** — ComboBox + FileDialog + mode-aware resolution picker |
| `CMakeLists.txt` | **MODIFY** — add `Qt6::QuickDialogs2`, new source files |
