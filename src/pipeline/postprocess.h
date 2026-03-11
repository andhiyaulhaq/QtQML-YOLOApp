#pragma once

#include "yolo_types.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class YoloPostProcessor {
public:
    YoloPostProcessor(MODEL_TYPE modelType, float rectConfidenceThreshold, float iouThreshold);

    void initBuffers(size_t strideNum); 

    void PostProcess(void* output, const std::vector<int64_t>& outputNodeDims, std::vector<DL_RESULT> &oResult, float resizeScales, const std::vector<std::string>& classes);

private:
    void greedyNMS(float iouThresh);

    MODEL_TYPE modelType;
    float rectConfidenceThreshold;
    float iouThreshold;

    // ── Two-Pass Postprocessing Buffers ──
    std::vector<float> m_bestScores;
    std::vector<int>   m_bestClassIds;
    std::vector<int>      m_classIds;
    std::vector<float>    m_confidences;
    std::vector<cv::Rect> m_boxes;
    std::vector<int>  m_nmsIndices;
    std::vector<int>  m_sortIndices;
    std::vector<bool> m_suppressed;
};
