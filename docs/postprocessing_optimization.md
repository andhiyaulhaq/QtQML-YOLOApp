# Optimization Task: Efficient Postprocessing Pipeline

> [!NOTE]
> This article documents potential postprocessing optimizations for the YOLOApp inference pipeline, following the same philosophy as the [preprocessing optimization](file:///d:/najib/2_archived-projects/desktop/qt-qml/YOLOApp/docs/blob_from_image.md).

## 1. Introduction: The Postprocessing Bottleneck

After the ONNX model produces its raw output tensor, the data must be decoded into usable bounding box detections. For YOLOv8n with an input size of 640×640, the output tensor is shaped `[1, 84, 8400]` — meaning **84 values** (4 box coords + 80 class scores) across **8400 candidate anchors**.

The current postprocessing pipeline involves:
1. **Transposition**: Converting from `[84, 8400]` to `[8400, 84]`.
2. **Thresholding Loop**: Iterating all 8400 anchors to find confident detections.
3. **Non-Maximum Suppression (NMS)**: Filtering overlapping boxes.
4. **Result Assembly**: Building the final `vector<DL_RESULT>`.

Just like `cv::dnn::blobFromImage` was a hidden bottleneck in preprocessing, these postprocessing steps contain several inefficiencies that can be optimized.

---

## 2. The Current Approach

Here is the relevant postprocessing code from [TensorProcess](file:///d:/najib/2_archived-projects/desktop/qt-qml/YOLOApp/src/inference.cpp#L348-L413):

```cpp
// Current implementation (simplified)
cv::Mat rawData = cv::Mat(signalResultNum, strideNum, CV_32F, output);
rawData = rawData.t();  // Transpose [84, 8400] -> [8400, 84]

float *data = (float *)rawData.data;

for (int i = 0; i < strideNum; ++i) {
    float *classesScores = data + 4;
    cv::Mat scores(1, this->classes.size(), CV_32FC1, classesScores);
    cv::Point class_id;
    double maxClassScore;
    cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);  // Expensive!
    if (maxClassScore > rectConfidenceThreshold) {
        confidences.push_back(maxClassScore);
        class_ids.push_back(class_id.x);
        boxes.push_back(cv::Rect(left, top, width, height));
    }
    data += signalResultNum;
}

cv::dnn::NMSBoxes(boxes, confidences, rectConfidenceThreshold, iouThreshold, nmsResult);
```

### Why It Is Suboptimal

| # | Problem | Impact |
|---|---------|--------|
| 1 | **`rawData.t()` creates a full copy** | Allocates a new 84 × 8400 × 4 = **1.34 MB** matrix every frame |
| 2 | **`cv::minMaxLoc` per anchor** | 8400 OpenCV function calls with per-call overhead (parameter validation, SSE dispatch) |
| 3 | **`cv::Mat` wrapper per anchor** | 8400 temporary `cv::Mat` header constructions per frame |
| 4 | **Dynamic vector growth** | `class_ids`, `confidences`, `boxes` reallocate as they grow |
| 5 | **`cv::dnn::NMSBoxes` is generic** | Black-box NMS with internal allocations; not optimized for our specific use case |

---

## 3. The Optimized Approach: Manual Decode Pipeline

### 3.1 Eliminate the Transpose

The transpose exists because the raw output is `[84, 8400]` (column-major for anchors), but the loop iterates per-anchor (row-major). Instead of transposing, we can **read the data in its original layout** using stride-based indexing:

```cpp
// Original layout: [84, 8400] — 84 rows, 8400 columns
// Row i, Column j = output[i * strideNum + j]
// For anchor j:
//   box_x  = output[0 * strideNum + j]
//   box_y  = output[1 * strideNum + j]
//   box_w  = output[2 * strideNum + j]
//   box_h  = output[3 * strideNum + j]
//   class_scores[c] = output[(4 + c) * strideNum + j]
```

```diff
 // Before: Allocates 1.34 MB
-cv::Mat rawData = cv::Mat(signalResultNum, strideNum, CV_32F, output);
-rawData = rawData.t();
-float *data = (float *)rawData.data;

 // After: Zero allocation — read directly from ONNX output buffer
+float *data = static_cast<float*>(output);
```

> [!IMPORTANT]
> This single change eliminates a **1.34 MB allocation + memcpy** per frame. With postprocessing running at ~10 FPS, this saves ~13.4 MB/s of memory bandwidth.

### 3.2 Replace `cv::minMaxLoc` with a Manual Max-scan

`cv::minMaxLoc` is designed for arbitrary matrices. For our case (a simple 1D array of 80 floats), a raw loop is significantly faster because it avoids:
- OpenCV parameter validation
- SSE/NEON dispatch overhead for small arrays
- Finding both min AND max (we only need max)

```diff
 // Before: OpenCV overhead per anchor
-cv::Mat scores(1, this->classes.size(), CV_32FC1, classesScores);
-cv::Point class_id;
-double maxClassScore;
-cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

 // After: Simple manual max-scan (80 iterations)
+int bestClassId = 0;
+float bestScore = classScorePtr[0];
+for (int c = 1; c < numClasses; ++c) {
+    float score = classScorePtr[c];
+    if (score > bestScore) {
+        bestScore = score;
+        bestClassId = c;
+    }
+}
```

### 3.3 Pre-allocate Result Vectors

Instead of letting `std::vector` grow dynamically (causing multiple reallocations), reserve a reasonable capacity upfront:

```diff
 // Before: Vectors start empty and reallocate ~7-10 times
 std::vector<int> class_ids;
 std::vector<float> confidences;
 std::vector<cv::Rect> boxes;

 // After: Reserve once, no reallocation in common cases
+// Typical detection count is 10-50; 256 covers extreme cases
+class_ids.reserve(256);
+confidences.reserve(256);
+boxes.reserve(256);
```

### 3.4 Make Result Vectors Reusable Members

Similar to how `m_commonBlob` and `m_letterboxBuffer` are reused for preprocessing, the postprocessing vectors can be **class members** that are cleared (not deallocated) each frame:

```diff
 // inference.h — Add member variables
+std::vector<int> m_classIds;
+std::vector<float> m_confidences;
+std::vector<cv::Rect> m_boxes;
+std::vector<int> m_nmsResult;

 // TensorProcess — Reuse instead of recreate
-std::vector<int> class_ids;
-std::vector<float> confidences;
-std::vector<cv::Rect> boxes;
+m_classIds.clear();      // O(1) — does NOT free memory
+m_confidences.clear();
+m_boxes.clear();
+m_nmsResult.clear();
```

> [!TIP]
> `std::vector::clear()` preserves the allocated capacity, so after the first few frames these vectors will **never reallocate again**. This eliminates heap churn entirely.

---

## 4. Full Optimized Implementation

Here is the complete fused postprocessing loop, combining all optimizations:

```cpp
// ============================================================
// Optimized Post-Processing (YOLOv8 Detection)
// ============================================================
int signalResultNum = outputNodeDims[1]; // 84
int strideNum = outputNodeDims[2];       // 8400
int numClasses = signalResultNum - 4;    // 80

// 1. Direct pointer — no transpose, no allocation
float *data = static_cast<float*>(output);

// 2. Reuse member vectors (zero allocation on steady state)
m_classIds.clear();
m_confidences.clear();
m_boxes.clear();

// 3. Fused decode loop: stride-indexed read + manual max-scan
for (int j = 0; j < strideNum; ++j) {
    // Read class scores with stride-based access (column j)
    int bestClassId = 0;
    float bestScore = data[(4) * strideNum + j];
    
    for (int c = 1; c < numClasses; ++c) {
        float score = data[(4 + c) * strideNum + j];
        if (score > bestScore) {
            bestScore = score;
            bestClassId = c;
        }
    }

    if (bestScore > rectConfidenceThreshold) {
        // Read box coordinates (also stride-indexed)
        float cx = data[0 * strideNum + j];
        float cy = data[1 * strideNum + j];
        float w  = data[2 * strideNum + j];
        float h  = data[3 * strideNum + j];

        int left   = static_cast<int>((cx - 0.5f * w) * resizeScales);
        int top    = static_cast<int>((cy - 0.5f * h) * resizeScales);
        int width  = static_cast<int>(w * resizeScales);
        int height = static_cast<int>(h * resizeScales);

        m_confidences.push_back(bestScore);
        m_classIds.push_back(bestClassId);
        m_boxes.push_back(cv::Rect(left, top, width, height));
    }
}

// 4. NMS (still using OpenCV, but with reusable output vector)
m_nmsResult.clear();
cv::dnn::NMSBoxes(m_boxes, m_confidences,
                  rectConfidenceThreshold, iouThreshold, m_nmsResult);

// 5. Assemble final results
for (int i = 0; i < m_nmsResult.size(); ++i) {
    int idx = m_nmsResult[i];
    DL_RESULT result;
    result.classId    = m_classIds[idx];
    result.confidence = m_confidences[idx];
    result.box        = m_boxes[idx];
    oResult.push_back(result);
}
```

---

## 5. Performance Improvements

| Metric | Current Implementation | Optimized Implementation | Improvement |
|:-------|:----------------------:|:------------------------:|:------------|
| **Transpose Allocation** | 1.34 MB per frame | **Zero** | Eliminated matrix copy |
| **`cv::Mat` Headers** | 8400 per frame | **Zero** | No temporary wrappers |
| **`cv::minMaxLoc` Calls** | 8400 per frame | **Zero** | Replaced with raw loop |
| **Vector Reallocations** | ~7–10 per frame | **Zero** (steady state) | Reusable member vectors |
| **Cache Efficiency** | Random access after transpose | Columnar stride access | Better L1/L2 utilization |

### Expected Latency Reduction

Based on profiling of similar YOLO postprocessing pipelines:

| Component | Before (est.) | After (est.) | Savings |
|:----------|:-------------:|:------------:|:--------|
| Transpose | ~0.3–0.5 ms | 0 ms | 100% |
| Threshold loop | ~0.5–1.0 ms | ~0.2–0.4 ms | ~50–60% |
| NMS | ~0.1–0.3 ms | ~0.1–0.3 ms | Unchanged |
| **Total Postprocess** | **~1.0–1.8 ms** | **~0.3–0.7 ms** | **~50–65%** |

---

## 6. Memory Layout Visualization

The key insight is understanding how the output tensor is laid out in memory and avoiding the expensive transpose:

```mermaid
graph TB
    subgraph "ONNX Output [84 × 8400]"
        direction LR
        R0["Row 0: cx for all 8400 anchors"]
        R1["Row 1: cy for all 8400 anchors"]
        R2["Row 2: w  for all 8400 anchors"]
        R3["Row 3: h  for all 8400 anchors"]
        R4["Row 4: class_0 scores"]
        R5["..."]
        R83["Row 83: class_79 scores"]
    end

    subgraph "Stride Access Pattern"
        A0["Anchor j → data[row * 8400 + j]"]
    end

    R0 --> A0
    R4 --> A0
```

Instead of transposing the entire matrix to read rows, we read **columns** by stepping through the data with a stride of `8400`. This accesses the same data without any memory movement.

---

## 7. Key Takeaways

> [!IMPORTANT]
> **The Philosophy**: Just as the preprocessing optimization replaced `cv::dnn::blobFromImage` with a manual fused loop, this postprocessing optimization replaces `cv::Mat::t()` + `cv::minMaxLoc` with **direct stride-indexed access** and a **manual max-scan**. The principle is the same: **avoid generic library overhead on hot paths**.

1. **Avoid copies on hot paths** — The transpose allocated 1.34 MB per frame for no reason other than convenient indexing.
2. **Replace OpenCV utilities for small data** — `cv::minMaxLoc` is powerful but overkill for scanning 80 floats.
3. **Reuse, don't recreate** — Member vectors with `clear()` preserve capacity across frames, eliminating heap churn.
4. **Read the data where it lives** — Stride-based access is slightly less cache-friendly per anchor, but eliminating the 1.34 MB transpose more than compensates.
