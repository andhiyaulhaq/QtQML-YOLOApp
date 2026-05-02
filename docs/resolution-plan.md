# Implementation Plan: Dynamic Camera Resolution Selection

**Date:** 2026-05-02
**Author:** Elite Software Engineer / System Architect

## Goal
Enable users to query and select supported camera resolutions dynamically from the UI, allowing them to choose between performance (VGA/720p) and high-fidelity (1080p/4K) widescreen modes.

---

## Technical Analysis
Currently, the resolution is hardcoded as `640x480` in `AppConfig`. To support user selection, we must:
1. **Discover Capabilities**: Query hardware for supported resolutions via Qt Multimedia.
2. **Dynamic Pipeline Reconfiguration**: Gracefully restart the OpenCV capture stream and reallocate UI buffers.
3. **Aspect Ratio Awareness**: Ensure YOLO preprocessing logic handles different aspect ratios (4:3 vs 16:9) without distortion.

---

## Memory Management & Leak Prevention Strategy
Processing high-resolution video (1080p) at 30+ FPS requires strict memory discipline to prevent application crashes and OS-level slowdowns.

### 1. Controlled Buffer Reallocation
*   **Problem**: Frequent allocation/deallocation of large image buffers causes heap fragmentation.
*   **Solution**: Use a **Pre-allocated Pool** (`cv::Mat m_framePool[3]`).
*   **Implementation**: When resolution changes, the `CaptureWorker` will stop the loop, call `.release()` on all mats, and then call `.create()` with the new dimensions. This ensures that memory is freed before new memory is requested.

### 2. Backpressure & Frame Dropping
*   **Problem**: If the AI inference is slower than the camera, frames will accumulate in the `QThread` event queue, leading to a memory "balloon."
*   **Solution**: Use an **Atomic Processing Flag**.
*   **Implementation**: The `CaptureWorker` checks `m_inferenceProcessingFlag->load()`. If true, it discards the current frame for inference purposes. This keeps memory usage constant regardless of system load.

### 3. GPU Memory Safety (Zero-Copy)
*   **Problem**: Passing frames to the UI often involves unnecessary copying from CPU to RAM to GPU.
*   **Solution**: **Mapped Buffer Re-use**.
*   **Implementation**: We use `QVideoFrame` with a pre-defined `QVideoFrameFormat`. We `map(WriteOnly)` the buffer, write pixel data directly via an OpenCV wrapper, and `unmap()` immediately. This prevents memory leaks in the GPU driver.

### 4. Smart Ownership
*   **Solution**: Use `std::shared_ptr<cv::Mat>` for inter-thread communication to ensure that a frame is only destroyed when both the Capture and Inference workers are finished with it.

---

## Data Flow & Coordinate Mapping
Changing the aspect ratio (e.g., from 4:3 to 16:9) introduces a "Spatial Mismatch" between the Camera and the AI Model (which is fixed at 640x640).

### 1. The Pre-Processing Flow (Letterboxing)
To maintain detection accuracy, we must avoid "stretching" the image.
*   **Step**: Instead of `cv::resize(original, resized, 640x640)`, we will implement a letterboxing function.
*   **Effect**: A 16:9 frame will be placed in the center of a 640x640 square with black bars at the top and bottom.

### 2. The Post-Processing Flow (Re-scaling)
The AI returns coordinates relative to the 640x640 square. These must be transformed back to the "World Space" of the selected resolution.
*   **Formula**:
    *   `x_actual = (x_model * 640 - padding_left) / scale_factor`
    *   `y_actual = (y_model * 640 - padding_top) / scale_factor`
*   **Implementation**: This logic must be centralized in the `PostProcessor` so that both the UI (Bounding Boxes) and the `CaptureWorker` (Segmentation Masks) see consistent, accurate coordinates.

---

## Proposed Changes

### 1. Capability Discovery Layer
#### [MODIFY] [VideoController.h](../app/src/core/VideoController.h)
*   Add `Q_PROPERTY(QVariantList supportedResolutions READ supportedResolutions NOTIFY supportedResolutionsChanged)`.
*   Add `Q_PROPERTY(QSize currentResolution READ currentResolution WRITE setCurrentResolution NOTIFY currentResolutionChanged)`.

### 2. Backend Logic: CaptureWorker
#### [MODIFY] [VideoController.cpp](../app/src/core/VideoController.cpp)
*   Update `CaptureWorker::startCapturing` to handle dynamic `QSize`.
*   Implement `m_capture.release()` and full object reset (`m_capture = cv::VideoCapture()`) before re-opening.
*   **Backend Switch**: Use `cv::CAP_MSMF` (Media Foundation) on Windows to ensure cleaner resource reclamation compared to `CAP_DSHOW`.
*   **Buffer Reuse**: Implement member buffers in `SegmentationPostProcessor` to reuse large matrices for mask processing, avoiding heap fragmentation.

### 3. UI Implementation
#### [MODIFY] [Main.qml](../app/content/Main.qml)
*   Add a `ComboBox` in the Metrics Panel for resolution selection.
*   Bind the model to `controller.supportedResolutions`.

---

## Verification Plan

### Automated Tests
*   **Leak Check**: Run the app for 10 minutes with frequent resolution switching; monitor `Private Bytes` in Windows Task Manager. It should remain stable.
*   **Throughput**: Verify that 1080p doesn't drop the UI frame rate below 24 FPS.

### Manual Verification
1. Start app at default resolution.
2. Switch to **1280x720 (720p)**. Verify the 16:9 aspect ratio changes correctly in the UI.
3. Switch between resolutions rapidly to verify stability.
4. Verify detection accuracy is maintained (no stretching of bounding boxes).

---

## Lessons Learned & Bug Resolutions

During the implementation phase, several critical "Elite" engineering challenges were addressed:

### 1. Blocked Event Loop Resolution
*   **Problem**: The `CaptureWorker` thread uses a tight `while(m_running)` loop for 30+ FPS capture. This blocks the Qt Event Loop, meaning standard `QueuedConnection` signals (like resolution updates) are never processed.
*   **Solution**: 
    1.  Introduced a `std::mutex` to protect `m_requestedResolution`.
    2.  Switched from `QueuedConnection` to `DirectConnection` for the `updateResolution` slot.
    3.  The GUI thread now writes directly to the worker's shared state, which the capture loop checks at the top of every iteration via an atomic flag.

### 2. Dynamic Coordinate Normalization
*   **Problem**: Bounding boxes appeared shifted or scaled incorrectly on 720p.
*   **Investigation**: Found that while `PostProcessor` was correctly re-mapping from 640x640 (AI) to 1280x720 (Camera), the `DetectionListModel` was still normalizing the results back to 0.0-1.0 using a hardcoded 640x480 constant.
*   **Solution**: Removed all hardcoded constants from `DetectionListModel`. It now accepts the actual `m_currentResolution` during every update to ensure normalized coordinates are always relative to the active camera frame.

### 3. Resolution Constraints
*   **Constraint**: The user interface was simplified to support only professional-grade 16:9 (720p) and standard 4:3 (480p) modes.
*   **Implementation**: Implemented a QSize filter in `refreshResolutions` to ignore intermediate or non-standard resolutions (e.g., 800x600, 1024x768) provided by the hardware driver, ensuring a curated user experience.
