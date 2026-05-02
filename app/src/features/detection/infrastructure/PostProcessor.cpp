#include "PostProcessor.h"
#include "SimdUtils.h"
#include <numeric>
#include <algorithm>
#include <iostream>

// ============================================================================
// DetectionPostProcessor
// ============================================================================

DetectionPostProcessor::DetectionPostProcessor(YoloTask::TaskType taskType, float rectConfidenceThreshold, float iouThreshold)
    : m_taskType(taskType), m_rectConfidenceThreshold(rectConfidenceThreshold), m_iouThreshold(iouThreshold) {}

void DetectionPostProcessor::initBuffers(size_t strideNum) {
    m_bestScores.resize(strideNum);
    m_bestClassIds.resize(strideNum);
    m_classIds.reserve(256);
    m_confidences.reserve(256);
    m_boxes.reserve(256);
    m_nmsIndices.reserve(64);
    m_sortIndices.reserve(256);
}

void DetectionPostProcessor::postProcess(void* output, const std::vector<int64_t>& outputNodeDims, std::vector<DetectionResult> &oResult, const LetterboxInfo& info, const std::vector<std::string>& classes, void* secondaryOutput, const std::vector<int64_t>& secondaryDims) {
    int signalResultNum = outputNodeDims[1]; 
    int strideNum = outputNodeDims[2];       
    int numClasses = signalResultNum - 4;    

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
        int mask = simd::check_threshold_sse41(bestS + j, m_rectConfidenceThreshold);
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

                int left   = static_cast<int>((cx - 0.5f * bw - info.padW) * info.scale);
                int top    = static_cast<int>((cy - 0.5f * bh - info.padH) * info.scale);
                int width  = static_cast<int>(bw * info.scale);
                int height = static_cast<int>(bh * info.scale);

                m_boxes.emplace_back(left, top, width, height);
            }
        }
    }

    for (; j < strideNum; ++j) {
        if (bestS[j] > m_rectConfidenceThreshold) {
            m_confidences.push_back(bestS[j]);
            m_classIds.push_back(bestC[j]);

            float cx = data[0 * strideNum + j];
            float cy = data[1 * strideNum + j];
            float bw = data[2 * strideNum + j];
            float bh = data[3 * strideNum + j];

            int left   = static_cast<int>((cx - 0.5f * bw - info.padW) * info.scale);
            int top    = static_cast<int>((cy - 0.5f * bh - info.padH) * info.scale);
            int width  = static_cast<int>(bw * info.scale);
            int height = static_cast<int>(bh * info.scale);

            m_boxes.emplace_back(left, top, width, height);
        }
    }

    m_nmsIndices.clear();
    greedyNMS(m_iouThreshold);

    for (size_t i = 0; i < m_nmsIndices.size(); ++i) {
        int idx = m_nmsIndices[i];
        DetectionResult result;
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

PosePostProcessor::PosePostProcessor(YoloTask::TaskType taskType, float rectConfidenceThreshold, float iouThreshold)
    : m_taskType(taskType), m_rectConfidenceThreshold(rectConfidenceThreshold), m_iouThreshold(iouThreshold) {}

void PosePostProcessor::initBuffers(size_t strideNum) {
    m_bestScores.resize(strideNum);
    m_bestClassIds.resize(strideNum);
    m_classIds.reserve(256);
    m_confidences.reserve(256);
    m_boxes.reserve(256);
    m_nmsIndices.reserve(64);
    m_sortIndices.reserve(256);
}

void PosePostProcessor::postProcess(void* output, const std::vector<int64_t>& outputNodeDims, std::vector<DetectionResult> &oResult, const LetterboxInfo& info, const std::vector<std::string>& classes, void* secondaryOutput, const std::vector<int64_t>& secondaryDims) {
    int strideNum = outputNodeDims[2];       

    float* data = static_cast<float*>(output);

    m_classIds.clear();
    m_confidences.clear();
    m_boxes.clear();
    m_keypoints.clear();

    int j = 0;
    for (; j <= strideNum - 4; j += 4) {
        int mask = simd::check_threshold_sse41(data + 4 * strideNum + j, m_rectConfidenceThreshold);
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

                int left   = static_cast<int>((cx - 0.5f * bw - info.padW) * info.scale);
                int top    = static_cast<int>((cy - 0.5f * bh - info.padH) * info.scale);
                int width  = static_cast<int>(bw * info.scale);
                int height = static_cast<int>(bh * info.scale);

                m_boxes.emplace_back(left, top, width, height);
                
                std::vector<cv::Point2f> kpts;
                for(int kp=0; kp<17; kp++) {
                    float kx = data[(5 + kp*3) * strideNum + idx];
                    float ky = data[(5 + kp*3 + 1) * strideNum + idx];
                    kpts.push_back(cv::Point2f((kx - info.padW) * info.scale, (ky - info.padH) * info.scale));
                }
                m_keypoints.push_back(kpts);
            }
        }
    }

    for (; j < strideNum; ++j) {
        float score = data[4 * strideNum + j];
        if (score > m_rectConfidenceThreshold) {
            m_confidences.push_back(score);
            m_classIds.push_back(0); 

            float cx = data[0 * strideNum + j];
            float cy = data[1 * strideNum + j];
            float bw = data[2 * strideNum + j];
            float bh = data[3 * strideNum + j];

            int left   = static_cast<int>((cx - 0.5f * bw - info.padW) * info.scale);
            int top    = static_cast<int>((cy - 0.5f * bh - info.padH) * info.scale);
            int width  = static_cast<int>(bw * info.scale);
            int height = static_cast<int>(bh * info.scale);

            m_boxes.emplace_back(left, top, width, height);
            
            std::vector<cv::Point2f> kpts;
            for(int k=0; k<17; k++) {
                float kx = data[(5 + k*3) * strideNum + j];
                float ky = data[(5 + k*3 + 1) * strideNum + j];
                kpts.push_back(cv::Point2f((kx - info.padW) * info.scale, (ky - info.padH) * info.scale));
            }
            m_keypoints.push_back(kpts);
        }
    }

    m_nmsIndices.clear();
    greedyNMS(m_iouThreshold);

    for (size_t i = 0; i < m_nmsIndices.size(); ++i) {
        int idx = m_nmsIndices[i];
        DetectionResult result;
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

SegmentationPostProcessor::SegmentationPostProcessor(YoloTask::TaskType taskType, float rectConfidenceThreshold, float iouThreshold)
    : m_taskType(taskType), m_rectConfidenceThreshold(rectConfidenceThreshold), m_iouThreshold(iouThreshold) {}

void SegmentationPostProcessor::initBuffers(size_t strideNum) {
    m_bestScores.resize(strideNum);
    m_bestClassIds.resize(strideNum);
    m_classIds.reserve(256);
    m_confidences.reserve(256);
    m_boxes.reserve(256);
    m_nmsIndices.reserve(64);
    m_sortIndices.reserve(256);
}

void SegmentationPostProcessor::postProcess(void* output, const std::vector<int64_t>& outputNodeDims, std::vector<DetectionResult> &oResult, const LetterboxInfo& info, const std::vector<std::string>& classes, void* secondaryOutput, const std::vector<int64_t>& secondaryDims) {
    if (!secondaryOutput) {
        std::cout << "[YOLO]: Segmentation requires secondary output tensor!" << std::endl;
        return;
    }

    int signalResultNum = outputNodeDims[1]; 
    int strideNum = outputNodeDims[2];       
    int numClasses = signalResultNum - 4 - 32; 

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

    int coeffOffset = 4 + numClasses; 

    int j = 0;
    for (; j <= strideNum - 4; j += 4) {
        int mask = simd::check_threshold_sse41(bestS + j, m_rectConfidenceThreshold);
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

                int left   = static_cast<int>((cx - 0.5f * bw - info.padW) * info.scale);
                int top    = static_cast<int>((cy - 0.5f * bh - info.padH) * info.scale);
                int width  = static_cast<int>(bw * info.scale);
                int height = static_cast<int>(bh * info.scale);

                m_boxes.emplace_back(left, top, width, height);

                std::vector<float> coeffs(32);
                for (int m = 0; m < 32; ++m) {
                    coeffs[m] = data[(coeffOffset + m) * strideNum + idx];
                }
                m_maskCoeffs.push_back(coeffs);
            }
        }
    }

    for (; j < strideNum; ++j) {
        if (bestS[j] > m_rectConfidenceThreshold) {
            m_confidences.push_back(bestS[j]);
            m_classIds.push_back(bestC[j]);

            float cx = data[0 * strideNum + j];
            float cy = data[1 * strideNum + j];
            float bw = data[2 * strideNum + j];
            float bh = data[3 * strideNum + j];

            int left   = static_cast<int>((cx - 0.5f * bw - info.padW) * info.scale);
            int top    = static_cast<int>((cy - 0.5f * bh - info.padH) * info.scale);
            int width  = static_cast<int>(bw * info.scale);
            int height = static_cast<int>(bh * info.scale);

            m_boxes.emplace_back(left, top, width, height);

            std::vector<float> coeffs(32);
            for (int m = 0; m < 32; ++m) {
                coeffs[m] = data[(coeffOffset + m) * strideNum + j];
            }
            m_maskCoeffs.push_back(coeffs);
        }
    }

    m_nmsIndices.clear();
    greedyNMS(m_iouThreshold);

    if (!m_nmsIndices.empty() && secondaryOutput != nullptr) {
        int maskChannels = secondaryDims[1]; 
        int maskH = secondaryDims[2];        
        int maskW = secondaryDims[3];        

        float* protoData = static_cast<float*>(secondaryOutput);
        m_protoMat = cv::Mat(maskChannels, maskH * maskW, CV_32F, protoData);

        for (size_t i = 0; i < m_nmsIndices.size(); ++i) {
            int idx = m_nmsIndices[i];
            DetectionResult result;
            result.classId    = m_classIds[idx];
            result.confidence = m_confidences[idx];
            result.box        = m_boxes[idx];

            cv::Mat coeffMat(1, maskChannels, CV_32F, m_maskCoeffs[idx].data());
            m_maskMat = coeffMat * m_protoMat; 

            cv::exp(-m_maskMat, m_maskMat);
            m_maskMat = 1.0 / (1.0 + m_maskMat);
            m_maskMat = m_maskMat.reshape(1, maskH); 

            cv::resize(m_maskMat, m_maskResized, cv::Size(maskW * 4, maskH * 4)); 
            
            cv::Rect validRoi(info.padW, info.padH, maskW * 4 - 2 * info.padW, maskH * 4 - 2 * info.padH);
            validRoi = validRoi & cv::Rect(0, 0, m_maskResized.cols, m_maskResized.rows);
            if (validRoi.width > 0 && validRoi.height > 0) {
                cv::Mat maskUnpadded = m_maskResized(validRoi);
                cv::resize(maskUnpadded, m_maskResized, cv::Size(maskUnpadded.cols * info.scale, maskUnpadded.rows * info.scale));
            }

            cv::Rect clipBox = result.box & cv::Rect(0, 0, m_maskResized.cols, m_maskResized.rows);
            if (clipBox.width > 0 && clipBox.height > 0) {
                cv::Mat roiMask = m_maskResized(clipBox) > 0.5f; 
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
