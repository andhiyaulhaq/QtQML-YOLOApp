#include "PostProcessor.h"
#include "SimdUtils.h"
#include <numeric>
#include <algorithm>
#include <iostream>

// ============================================================================
// DetectionPostProcessor
// ============================================================================

DetectionPostProcessor::DetectionPostProcessor(MODEL_TYPE modelType, float rectConfidenceThreshold, float iouThreshold)
    : modelType(modelType), rectConfidenceThreshold(rectConfidenceThreshold), iouThreshold(iouThreshold) {}

void DetectionPostProcessor::initBuffers(size_t strideNum) {
    m_bestScores.resize(strideNum);
    m_bestClassIds.resize(strideNum);
    m_classIds.reserve(256);
    m_confidences.reserve(256);
    m_boxes.reserve(256);
    m_nmsIndices.reserve(64);
    m_sortIndices.reserve(256);
}

void DetectionPostProcessor::PostProcess(void* output, const std::vector<int64_t>& outputNodeDims, std::vector<DL_RESULT> &oResult, float resizeScales, const std::vector<std::string>& classes, void* secondaryOutput, const std::vector<int64_t>& secondaryDims) {
    int signalResultNum = outputNodeDims[1]; // 84
    int strideNum = outputNodeDims[2];       // 8400
    int numClasses = signalResultNum - 4;    // 80

    float* data = static_cast<float*>(output);

    float* row0 = data + 4 * strideNum;
    memcpy(m_bestScores.data(), row0, strideNum * sizeof(float));
    memset(m_bestClassIds.data(), 0, strideNum * sizeof(int));

        for (int c = 1; c < numClasses; ++c) {
        simd::update_best_scores_sse41(data + (4 + c) * strideNum, m_bestScores.data(), m_bestClassIds.data(), c, strideNum);
    }

    m_classIds.clear();
    m_confidences.clear();
    m_boxes.clear();

    const float* bestS = m_bestScores.data();
    const int*   bestC = m_bestClassIds.data();

    int j = 0;
    for (; j <= strideNum - 4; j += 4) {
        int mask = simd::check_threshold_sse41(bestS + j, rectConfidenceThreshold);
        if (mask == 0) continue; // Skip all 4

        for (int k = 0; k < 4; ++k) {
            if (mask & (1 << k)) {
                int idx = j + k;
                m_confidences.push_back(bestS[idx]);
                m_classIds.push_back(bestC[idx]);

                float cx = data[0 * strideNum + idx];
                float cy = data[1 * strideNum + idx];
                float bw = data[2 * strideNum + idx];
                float bh = data[3 * strideNum + idx];

                int left   = static_cast<int>((cx - 0.5f * bw) * resizeScales);
                int top    = static_cast<int>((cy - 0.5f * bh) * resizeScales);
                int width  = static_cast<int>(bw * resizeScales);
                int height = static_cast<int>(bh * resizeScales);

                m_boxes.emplace_back(left, top, width, height);
            }
        }
    }

    // Tail
    for (; j < strideNum; ++j) {
        if (bestS[j] > rectConfidenceThreshold) {
            m_confidences.push_back(bestS[j]);
            m_classIds.push_back(bestC[j]);

            float cx = data[0 * strideNum + j];
            float cy = data[1 * strideNum + j];
            float bw = data[2 * strideNum + j];
            float bh = data[3 * strideNum + j];

            int left   = static_cast<int>((cx - 0.5f * bw) * resizeScales);
            int top    = static_cast<int>((cy - 0.5f * bh) * resizeScales);
            int width  = static_cast<int>(bw * resizeScales);
            int height = static_cast<int>(bh * resizeScales);

            m_boxes.emplace_back(left, top, width, height);
        }
    }

    m_nmsIndices.clear();
    greedyNMS(iouThreshold);

    for (size_t i = 0; i < m_nmsIndices.size(); ++i) {
        int idx = m_nmsIndices[i];
        DL_RESULT result;
        result.classId    = m_classIds[idx];
        result.confidence = m_confidences[idx];
        result.box        = m_boxes[idx];
        oResult.push_back(result);
    }
}

void DetectionPostProcessor::greedyNMS(float iouThresh) {
    int n = static_cast<int>(m_confidences.size());
    if (n == 0) return;
    
    m_sortIndices.resize(n);
    std::iota(m_sortIndices.begin(), m_sortIndices.end(), 0);
    std::sort(m_sortIndices.begin(), m_sortIndices.end(),
              [this](int a, int b) {
                  return m_confidences[a] > m_confidences[b];
              });
    
    m_suppressed.assign(n, false);
    
    for (int i = 0; i < n; ++i) {
        int idx = m_sortIndices[i];
        if (m_suppressed[idx]) continue;    
        
        m_nmsIndices.push_back(idx);        
        
        const cv::Rect& a = m_boxes[idx];
        float areaA = static_cast<float>(a.width * a.height);
        
        for (int k = i + 1; k < n; ++k) {
            int kidx = m_sortIndices[k];
            if (m_suppressed[kidx]) continue;
            
            const cv::Rect& b = m_boxes[kidx];
            
            int x1 = std::max(a.x, b.x);
            int y1 = std::max(a.y, b.y);
            int x2 = std::min(a.x + a.width,  b.x + b.width);
            int y2 = std::min(a.y + a.height, b.y + b.height);
            
            if (x2 <= x1 || y2 <= y1) continue;
            
            float intersection = static_cast<float>((x2 - x1) * (y2 - y1));
            float areaB = static_cast<float>(b.width * b.height);
            float unionArea = areaA + areaB - intersection;
            float iou = intersection / unionArea;
            
            if (iou > iouThresh) {
                m_suppressed[kidx] = true;  
            }
        }
    }
}


// ============================================================================
// PosePostProcessor
// ============================================================================

PosePostProcessor::PosePostProcessor(MODEL_TYPE modelType, float rectConfidenceThreshold, float iouThreshold)
    : modelType(modelType), rectConfidenceThreshold(rectConfidenceThreshold), iouThreshold(iouThreshold) {}

void PosePostProcessor::initBuffers(size_t strideNum) {
    m_bestScores.resize(strideNum);
    m_bestClassIds.resize(strideNum);
    m_classIds.reserve(256);
    m_confidences.reserve(256);
    m_boxes.reserve(256);
    m_nmsIndices.reserve(64);
    m_sortIndices.reserve(256);
}

void PosePostProcessor::PostProcess(void* output, const std::vector<int64_t>& outputNodeDims, std::vector<DL_RESULT> &oResult, float resizeScales, const std::vector<std::string>& classes, void* secondaryOutput, const std::vector<int64_t>& secondaryDims) {
    // Basic implementation for Pose Estimation placeholder
    // In a full implementation, you'd extract the 17 keypoints from the tensor and store them in DL_RESULT::keyPoints
    int signalResultNum = outputNodeDims[1]; 
    int strideNum = outputNodeDims[2];       

    float* data = static_cast<float*>(output);

    // Usually Pose is 56 (4 box + 1 conf + 17*3 keypoints). Adjust based on exact tensor size!
    int numClasses = 1;

    m_classIds.clear();
    m_confidences.clear();
    m_boxes.clear();
    m_keypoints.clear();

    int j = 0;
    for (; j <= strideNum - 4; j += 4) {
        int mask = simd::check_threshold_sse41(data + 4 * strideNum + j, rectConfidenceThreshold);
        if (mask == 0) continue;

        for (int k = 0; k < 4; ++k) {
            if (mask & (1 << k)) {
                int idx = j + k;
                float score = data[4 * strideNum + idx];
                m_confidences.push_back(score);
                m_classIds.push_back(0);

                float cx = data[0 * strideNum + idx];
                float cy = data[1 * strideNum + idx];
                float bw = data[2 * strideNum + idx];
                float bh = data[3 * strideNum + idx];

                int left   = static_cast<int>((cx - 0.5f * bw) * resizeScales);
                int top    = static_cast<int>((cy - 0.5f * bh) * resizeScales);
                int width  = static_cast<int>(bw * resizeScales);
                int height = static_cast<int>(bh * resizeScales);

                m_boxes.emplace_back(left, top, width, height);
                
                std::vector<cv::Point2f> kpts;
                for(int kp=0; kp<17; kp++) {
                    float kx = data[(5 + kp*3) * strideNum + idx];
                    float ky = data[(5 + kp*3 + 1) * strideNum + idx];
                    kpts.push_back(cv::Point2f(kx * resizeScales, ky * resizeScales));
                }
                m_keypoints.push_back(kpts);
            }
        }
    }

    for (; j < strideNum; ++j) {
        float score = data[4 * strideNum + j];
        if (score > rectConfidenceThreshold) {
            m_confidences.push_back(score);
            m_classIds.push_back(0); 
// ...
            float cx = data[0 * strideNum + j];
            float cy = data[1 * strideNum + j];
            float bw = data[2 * strideNum + j];
            float bh = data[3 * strideNum + j];

            int left   = static_cast<int>((cx - 0.5f * bw) * resizeScales);
            int top    = static_cast<int>((cy - 0.5f * bh) * resizeScales);
            int width  = static_cast<int>(bw * resizeScales);
            int height = static_cast<int>(bh * resizeScales);

            m_boxes.emplace_back(left, top, width, height);
            
            std::vector<cv::Point2f> kpts;
            for(int k=0; k<17; k++) {
                float kx = data[(5 + k*3) * strideNum + j];
                float ky = data[(5 + k*3 + 1) * strideNum + j];
                kpts.push_back(cv::Point2f(kx * resizeScales, ky * resizeScales));
            }
            m_keypoints.push_back(kpts);
        }
    }

    m_nmsIndices.clear();
    greedyNMS(iouThreshold);

    for (size_t i = 0; i < m_nmsIndices.size(); ++i) {
        int idx = m_nmsIndices[i];
        DL_RESULT result;
        result.classId    = m_classIds[idx];
        result.confidence = m_confidences[idx];
        result.box        = m_boxes[idx];
        result.keyPoints  = m_keypoints[idx];
        oResult.push_back(result);
    }
}

void PosePostProcessor::greedyNMS(float iouThresh) {
    int n = static_cast<int>(m_confidences.size());
    if (n == 0) return;
    
    m_sortIndices.resize(n);
    std::iota(m_sortIndices.begin(), m_sortIndices.end(), 0);
    std::sort(m_sortIndices.begin(), m_sortIndices.end(),
              [this](int a, int b) {
                  return m_confidences[a] > m_confidences[b];
              });
    
    m_suppressed.assign(n, false);
    
    for (int i = 0; i < n; ++i) {
        int idx = m_sortIndices[i];
        if (m_suppressed[idx]) continue;    
        
        m_nmsIndices.push_back(idx);        
        
        const cv::Rect& a = m_boxes[idx];
        float areaA = static_cast<float>(a.width * a.height);
        
        for (int k = i + 1; k < n; ++k) {
            int kidx = m_sortIndices[k];
            if (m_suppressed[kidx]) continue;
            
            const cv::Rect& b = m_boxes[kidx];
            
            int x1 = std::max(a.x, b.x);
            int y1 = std::max(a.y, b.y);
            int x2 = std::min(a.x + a.width,  b.x + b.width);
            int y2 = std::min(a.y + a.height, b.y + b.height);
            
            if (x2 <= x1 || y2 <= y1) continue;
            
            float intersection = static_cast<float>((x2 - x1) * (y2 - y1));
            float areaB = static_cast<float>(b.width * b.height);
            float unionArea = areaA + areaB - intersection;
            float iou = intersection / unionArea;
            
            if (iou > iouThresh) {
                m_suppressed[kidx] = true;  
            }
        }
    }
}

// ============================================================================
// SegmentationPostProcessor
// ============================================================================

SegmentationPostProcessor::SegmentationPostProcessor(MODEL_TYPE modelType, float rectConfidenceThreshold, float iouThreshold)
    : modelType(modelType), rectConfidenceThreshold(rectConfidenceThreshold), iouThreshold(iouThreshold) {}

void SegmentationPostProcessor::initBuffers(size_t strideNum) {
    m_bestScores.resize(strideNum);
    m_bestClassIds.resize(strideNum);
    m_classIds.reserve(256);
    m_confidences.reserve(256);
    m_boxes.reserve(256);
    m_nmsIndices.reserve(64);
    m_sortIndices.reserve(256);
}

void SegmentationPostProcessor::PostProcess(void* output, const std::vector<int64_t>& outputNodeDims, std::vector<DL_RESULT> &oResult, float resizeScales, const std::vector<std::string>& classes, void* secondaryOutput, const std::vector<int64_t>& secondaryDims) {
    if (!secondaryOutput) {
        std::cout << "[YOLO]: Segmentation requires secondary output tensor!" << std::endl;
        return;
    }

    int signalResultNum = outputNodeDims[1]; // 116 (4 box + 80 class + 32 mask coeff)
    int strideNum = outputNodeDims[2];       // 8400
    int numClasses = signalResultNum - 4 - 32; // 80

    float* data = static_cast<float*>(output);

    float* row0 = data + 4 * strideNum;
    memcpy(m_bestScores.data(), row0, strideNum * sizeof(float));
    memset(m_bestClassIds.data(), 0, strideNum * sizeof(int));

        for (int c = 1; c < numClasses; ++c) {
        simd::update_best_scores_sse41(data + (4 + c) * strideNum, m_bestScores.data(), m_bestClassIds.data(), c, strideNum);
    }

    m_classIds.clear();
    m_confidences.clear();
    m_boxes.clear();
    m_maskCoeffs.clear();

    const float* bestS = m_bestScores.data();
    const int*   bestC = m_bestClassIds.data();

    // Box offset and Mask coeff offset
    int coeffOffset = 4 + numClasses; // 84

    int j = 0;
    for (; j <= strideNum - 4; j += 4) {
        int mask = simd::check_threshold_sse41(bestS + j, rectConfidenceThreshold);
        if (mask == 0) continue;

        for (int k = 0; k < 4; ++k) {
            if (mask & (1 << k)) {
                int idx = j + k;
                m_confidences.push_back(bestS[idx]);
                m_classIds.push_back(bestC[idx]);

                float cx = data[0 * strideNum + idx];
                float cy = data[1 * strideNum + idx];
                float bw = data[2 * strideNum + idx];
                float bh = data[3 * strideNum + idx];

                int left   = static_cast<int>((cx - 0.5f * bw) * resizeScales);
                int top    = static_cast<int>((cy - 0.5f * bh) * resizeScales);
                int width  = static_cast<int>(bw * resizeScales);
                int height = static_cast<int>(bh * resizeScales);

                m_boxes.emplace_back(left, top, width, height);

                // Extract 32 mask coefficients
                std::vector<float> coeffs(32);
                for (int m = 0; m < 32; ++m) {
                    coeffs[m] = data[(coeffOffset + m) * strideNum + idx];
                }
                m_maskCoeffs.push_back(coeffs);
            }
        }
    }

    for (; j < strideNum; ++j) {
        if (bestS[j] > rectConfidenceThreshold) {
            m_confidences.push_back(bestS[j]);
            m_classIds.push_back(bestC[j]);

            float cx = data[0 * strideNum + j];
            float cy = data[1 * strideNum + j];
            float bw = data[2 * strideNum + j];
            float bh = data[3 * strideNum + j];

            int left   = static_cast<int>((cx - 0.5f * bw) * resizeScales);
            int top    = static_cast<int>((cy - 0.5f * bh) * resizeScales);
            int width  = static_cast<int>(bw * resizeScales);
            int height = static_cast<int>(bh * resizeScales);

            m_boxes.emplace_back(left, top, width, height);

            // Extract 32 mask coefficients
            std::vector<float> coeffs(32);
            for (int m = 0; m < 32; ++m) {
                coeffs[m] = data[(coeffOffset + m) * strideNum + j];
            }
            m_maskCoeffs.push_back(coeffs);
        }
    }

    m_nmsIndices.clear();
    greedyNMS(iouThreshold);

    // If there are detections, process masks
    if (!m_nmsIndices.empty() && secondaryOutput != nullptr) {
        // Prototype Masks (typically [1, 32, 160, 160])
        int maskChannels = secondaryDims[1]; // 32
        int maskH = secondaryDims[2];        // 160
        int maskW = secondaryDims[3];        // 160

        float* protoData = static_cast<float*>(secondaryOutput);

        // Reshape to [32, 25600]
        cv::Mat protoMat(maskChannels, maskH * maskW, CV_32F, protoData);

        for (size_t i = 0; i < m_nmsIndices.size(); ++i) {
            int idx = m_nmsIndices[i];
            DL_RESULT result;
            result.classId    = m_classIds[idx];
            result.confidence = m_confidences[idx];
            result.box        = m_boxes[idx];

            // Matrix multiplication for the mask
            cv::Mat coeffMat(1, maskChannels, CV_32F, m_maskCoeffs[idx].data());
            cv::Mat maskMat = coeffMat * protoMat; // [1, 25600]

            // Sigmoid and reshape
            cv::exp(-maskMat, maskMat);
            maskMat = 1.0 / (1.0 + maskMat);
            maskMat = maskMat.reshape(1, maskH); // [160, 160]

            // Resize the mask up to the original image coordinates to crop via bounding box
            cv::Mat maskResized;
            cv::resize(maskMat, maskResized, cv::Size(maskW * 4 * resizeScales, maskH * 4 * resizeScales)); // Usually 4x stride

            // Crop mask to bounding box bounds 
            // Clamp rect to mask bounds
            cv::Rect clipBox = result.box & cv::Rect(0, 0, maskResized.cols, maskResized.rows);
            if (clipBox.width > 0 && clipBox.height > 0) {
                cv::Mat roiMask = maskResized(clipBox) > 0.5f; // Threshold
                result.boxMask = roiMask.clone();
            } else {
                result.boxMask = cv::Mat();
            }

            oResult.push_back(result);
        }
    }
}

void SegmentationPostProcessor::greedyNMS(float iouThresh) {
    int n = static_cast<int>(m_confidences.size());
    if (n == 0) return;
    
    m_sortIndices.resize(n);
    std::iota(m_sortIndices.begin(), m_sortIndices.end(), 0);
    std::sort(m_sortIndices.begin(), m_sortIndices.end(),
              [this](int a, int b) {
                  return m_confidences[a] > m_confidences[b];
              });
    
    m_suppressed.assign(n, false);
    
    for (int i = 0; i < n; ++i) {
        int idx = m_sortIndices[i];
        if (m_suppressed[idx]) continue;    
        
        m_nmsIndices.push_back(idx);        
        
        const cv::Rect& a = m_boxes[idx];
        float areaA = static_cast<float>(a.width * a.height);
        
        for (int k = i + 1; k < n; ++k) {
            int kidx = m_sortIndices[k];
            if (m_suppressed[kidx]) continue;
            
            const cv::Rect& b = m_boxes[kidx];
            
            int x1 = std::max(a.x, b.x);
            int y1 = std::max(a.y, b.y);
            int x2 = std::min(a.x + a.width,  b.x + b.width);
            int y2 = std::min(a.y + a.height, b.y + b.height);
            
            if (x2 <= x1 || y2 <= y1) continue;
            
            float intersection = static_cast<float>((x2 - x1) * (y2 - y1));
            float areaB = static_cast<float>(b.width * b.height);
            float unionArea = areaA + areaB - intersection;
            float iou = intersection / unionArea;
            
            if (iou > iouThresh) {
                m_suppressed[kidx] = true;  
            }
        }
    }
}
