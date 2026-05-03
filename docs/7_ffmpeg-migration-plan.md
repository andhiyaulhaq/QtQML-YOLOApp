# 7. FFmpeg Native Migration Plan

## Status: COMPLETED

Transitioning from OpenCV's `cv::VideoCapture` to the native FFmpeg C++ libraries (`libavcodec`, `libavformat`, etc.) has been successfully implemented, resolving performance bottlenecks and orientation issues.

---

## Final Implementation Approach

### 1. Surgical Stream Control & Accurate Seeking
To ensure smooth and accurate scrubbing (GOP-aware), the `FFmpegVideoSource::seekToFrame` method implements a two-step process:
- **Keyframe Seek**: Uses `avformat_seek_file` with `AVSEEK_FLAG_BACKWARD` to jump to the nearest keyframe before the target.
- **Decoding Catch-up**: Sequentially decodes and discards frames from the keyframe until the exact target frame index is reached. This eliminates "lag" and ensures frame-accurate display.

### 2. Precise Frame Index Synchronization
To prevent temporal drift and ensure the progress tracker reaches the end accurately:
- **PTS-Based Tracking**: The `m_currentFrameIndex` is synchronized using the Presentation Timestamp (PTS) of each decoded frame.
- **AVRational Rescaling**: `av_rescale_q` is used with `av_inv_q(stream->avg_frame_rate)` to convert timestamps to frame indices without floating-point precision loss.
- **Duration-Based Math**: Total frame count is calculated from the format context duration when `nb_frames` is unavailable or unreliable.

### 3. Robust Rotation & Orientation Handling
Portrait videos (common in mobile captures) are handled via metadata inspection:
- **Side Data Extraction**: Rotation is detected by reading `AV_PKT_DATA_DISPLAYMATRIX` from `codecpar->coded_side_data` (FFmpeg 7.0+ compliant).
- **Dynamic Transformation**: `cv::rotate` is applied (90/180/270 CCW/CW) within the frame reading path.
- **Resolution Awareness**: The `currentResolution()` method automatically swaps dimensions if the video is vertical.

### 4. Decoder Flushing (EOF Management)
To ensure every single frame is retrieved (even those stuck in the decoder's B-frame buffer):
- **Null Packet Signal**: When `av_read_frame` returns EOF, a `nullptr` packet is sent to the decoder to trigger flushing mode.
- **Drain Loop**: `readFrameInternal` continues to drain the decoder until all buffered frames are processed, preventing premature video termination.

### 5. CaptureWorker Pacing Logic
The `CaptureWorker` was refactored to use a unified `ICaptureSource` interface:
- **Real-time Sync**: Pacing is calculated using `nativeFps()` and high-resolution system clocks.
- **Sync Point Reset**: After a seek, the `m_videoStartTime` and `m_startFrameIndex` are reset to the seek target, ensuring the video resumes at exactly 1x speed.

---

## Result Achieved
The application now features professional-grade media playback:
- **Instant Scrubber**: Fluid seeking with zero "stale" frames.
- **Perfect Loop**: Frame-accurate video ending and looping.
- **Native Orientation**: Correct rendering for both portrait and landscape mobile videos.
- **Low Overhead**: Optimized memory management with reusable buffers and efficient colorspace conversion.
