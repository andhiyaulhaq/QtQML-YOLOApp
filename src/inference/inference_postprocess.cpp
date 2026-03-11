#include "inference.h"
#include <numeric>
#include <algorithm>
#include <iostream>

void YOLO_V8::PostProcess(void* output, const std::vector<int64_t>& outputNodeDims, std::vector<DL_RESULT> &oResult) {
  switch (modelType) {
  case YOLO_DETECT_V8:
  case YOLO_DETECT_V8_HALF: {
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

    break;
  }
  case YOLO_CLS:
  case YOLO_CLS_HALF: {
    cv::Mat rawData;
    if (modelType == YOLO_CLS) {
      rawData = cv::Mat(1, this->classes.size(), CV_32F, output);
    } else {
      rawData = cv::Mat(1, this->classes.size(), CV_16F, output);
      rawData.convertTo(rawData, CV_32F);
    }
    float *data = (float *)rawData.data;

    DL_RESULT result;
    for (int i = 0; i < this->classes.size(); i++) {
      result.classId = i;
      result.confidence = data[i];
      oResult.push_back(result);
    }
    break;
  }
  default:
    std::cout << "[YOLO_V8]: " << "Not support model type." << std::endl;
  }
}

void YOLO_V8::greedyNMS(float iouThresh) {
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
