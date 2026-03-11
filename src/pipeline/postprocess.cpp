#include "postprocess.h"
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

    float* data;
    cv::Mat rawData; 
    if (modelType == YOLO_DETECT_V8) {
        data = static_cast<float*>(output);  
    } else {
        rawData = cv::Mat(signalResultNum, strideNum, CV_16F, output);
        rawData.convertTo(rawData, CV_32F);  
        data = reinterpret_cast<float*>(rawData.data);
    }

    float* row0 = data + 4 * strideNum;
    memcpy(m_bestScores.data(), row0, strideNum * sizeof(float));
    memset(m_bestClassIds.data(), 0, strideNum * sizeof(int));

    for (int c = 1; c < numClasses; ++c) {
        const float* rowC = data + (4 + c) * strideNum;
        float* bestS = m_bestScores.data();
        int*   bestC = m_bestClassIds.data();

        for (int j = 0; j < strideNum; ++j) {
            if (rowC[j] > bestS[j]) {
                bestS[j] = rowC[j];
                bestC[j] = c;
            }
        }
    }

    m_classIds.clear();
    m_confidences.clear();
    m_boxes.clear();

    const float* bestS = m_bestScores.data();
    const int*   bestC = m_bestClassIds.data();

    for (int j = 0; j < strideNum; ++j) {
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
// ClassificationPostProcessor
// ============================================================================

ClassificationPostProcessor::ClassificationPostProcessor(MODEL_TYPE modelType) : modelType(modelType) {}

void ClassificationPostProcessor::PostProcess(void* output, const std::vector<int64_t>& outputNodeDims, std::vector<DL_RESULT> &oResult, float resizeScales, const std::vector<std::string>& classes, void* secondaryOutput, const std::vector<int64_t>& secondaryDims) {
    cv::Mat rawData;
    if (modelType == YOLO_CLS) {
      rawData = cv::Mat(1, classes.size(), CV_32F, output);
    } else {
      rawData = cv::Mat(1, classes.size(), CV_16F, output);
      rawData.convertTo(rawData, CV_32F);
    }
    float *data = (float *)rawData.data;

    DL_RESULT result;
    for (int i = 0; i < classes.size(); i++) {
      result.classId = i;
      result.confidence = data[i];
      oResult.push_back(result);
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

    float* data;
    cv::Mat rawData; 
    if (modelType == YOLO_POSE) {
        data = static_cast<float*>(output);  
    } else {
        rawData = cv::Mat(signalResultNum, strideNum, CV_16F, output);
        rawData.convertTo(rawData, CV_32F);  
        data = reinterpret_cast<float*>(rawData.data);
    }

    // Usually Pose is 56 (4 box + 1 conf + 17*3 keypoints). Adjust based on exact tensor size!
    int numClasses = 1;

    m_classIds.clear();
    m_confidences.clear();
    m_boxes.clear();
    m_keypoints.clear();

    for (int j = 0; j < strideNum; ++j) {
        float score = data[4 * strideNum + j];
        if (score > rectConfidenceThreshold) {
            m_confidences.push_back(score);
            m_classIds.push_back(0); // Pose only has 1 class (Person)

            float cx = data[0 * strideNum + j];
            float cy = data[1 * strideNum + j];
            float bw = data[2 * strideNum + j];
            float bh = data[3 * strideNum + j];

            int left   = static_cast<int>((cx - 0.5f * bw) * resizeScales);
            int top    = static_cast<int>((cy - 0.5f * bh) * resizeScales);
            int width  = static_cast<int>(bw * resizeScales);
            int height = static_cast<int>(bh * resizeScales);

            m_boxes.emplace_back(left, top, width, height);
            
            // Extract 17 keypoints (x, y, conf)
            std::vector<cv::Point2f> kpts;
            for(int k=0; k<17; k++) {
                float kx = data[(5 + k*3) * strideNum + j];
                float ky = data[(5 + k*3 + 1) * strideNum + j];
                float kconf = data[(5 + k*3 + 2) * strideNum + j];
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
        std::cout << "[YOLO_V8]: Segmentation requires secondary output tensor!" << std::endl;
        return;
    }

    int signalResultNum = outputNodeDims[1]; // 116 (4 box + 80 class + 32 mask coeff)
    int strideNum = outputNodeDims[2];       // 8400
    int numClasses = signalResultNum - 4 - 32; // 80

    float* data;
    cv::Mat rawData; 
    if (modelType == YOLO_SEG) {
        data = static_cast<float*>(output);  
    } else {
        rawData = cv::Mat(signalResultNum, strideNum, CV_16F, output);
        rawData.convertTo(rawData, CV_32F);  
        data = reinterpret_cast<float*>(rawData.data);
    }

    float* row0 = data + 4 * strideNum;
    memcpy(m_bestScores.data(), row0, strideNum * sizeof(float));
    memset(m_bestClassIds.data(), 0, strideNum * sizeof(int));

    for (int c = 1; c < numClasses; ++c) {
        const float* rowC = data + (4 + c) * strideNum;
        float* bestS = m_bestScores.data();
        int*   bestC = m_bestClassIds.data();

        for (int j = 0; j < strideNum; ++j) {
            if (rowC[j] > bestS[j]) {
                bestS[j] = rowC[j];
                bestC[j] = c;
            }
        }
    }

    m_classIds.clear();
    m_confidences.clear();
    m_boxes.clear();
    m_maskCoeffs.clear();

    const float* bestS = m_bestScores.data();
    const int*   bestC = m_bestClassIds.data();

    // Box offset and Mask coeff offset
    int coeffOffset = 4 + numClasses; // 84

    for (int j = 0; j < strideNum; ++j) {
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

        float* protoData;
        cv::Mat rawProtoData;
        if (modelType == YOLO_SEG) {
            protoData = static_cast<float*>(secondaryOutput);
        } else {
            rawProtoData = cv::Mat(maskChannels, maskH * maskW, CV_16F, secondaryOutput);
            rawProtoData.convertTo(rawProtoData, CV_32F);
            protoData = reinterpret_cast<float*>(rawProtoData.data);
        }

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
