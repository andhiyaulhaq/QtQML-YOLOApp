#pragma once

#include "yolo_types.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class IPostProcessor {
public:
    virtual ~IPostProcessor() = default;
    virtual void initBuffers(size_t strideNum) = 0;
    virtual void PostProcess(void* output, const std::vector<int64_t>& outputNodeDims, std::vector<DL_RESULT>& oResult, float resizeScales, const std::vector<std::string>& classes, void* secondaryOutput = nullptr, const std::vector<int64_t>& secondaryDims = {}) = 0;
};

class DetectionPostProcessor : public IPostProcessor {
public:
    DetectionPostProcessor(MODEL_TYPE modelType, float rectConfidenceThreshold, float iouThreshold);
    void initBuffers(size_t strideNum) override;
    void PostProcess(void* output, const std::vector<int64_t>& outputNodeDims, std::vector<DL_RESULT>& oResult, float resizeScales, const std::vector<std::string>& classes, void* secondaryOutput = nullptr, const std::vector<int64_t>& secondaryDims = {}) override;
private:
    void greedyNMS(float iouThresh);
    MODEL_TYPE modelType;
    float rectConfidenceThreshold;
    float iouThreshold;
    std::vector<float> m_bestScores;
    std::vector<int> m_bestClassIds;
    std::vector<int> m_classIds;
    std::vector<float> m_confidences;
    std::vector<cv::Rect> m_boxes;
    std::vector<int> m_nmsIndices;
    std::vector<int> m_sortIndices;
    std::vector<bool> m_suppressed;
};

class PosePostProcessor : public IPostProcessor {
public:
    PosePostProcessor(MODEL_TYPE modelType, float rectConfidenceThreshold, float iouThreshold);
    void initBuffers(size_t strideNum) override;
    void PostProcess(void* output, const std::vector<int64_t>& outputNodeDims, std::vector<DL_RESULT>& oResult, float resizeScales, const std::vector<std::string>& classes, void* secondaryOutput = nullptr, const std::vector<int64_t>& secondaryDims = {}) override;
private:
    void greedyNMS(float iouThresh);
    MODEL_TYPE modelType;
    float rectConfidenceThreshold;
    float iouThreshold;
    std::vector<float> m_bestScores;
    std::vector<int> m_bestClassIds;
    std::vector<int> m_classIds;
    std::vector<float> m_confidences;
    std::vector<cv::Rect> m_boxes;
    std::vector<std::vector<cv::Point2f>> m_keypoints;
    std::vector<int> m_nmsIndices;
    std::vector<int> m_sortIndices;
    std::vector<bool> m_suppressed;
};

class SegmentationPostProcessor : public IPostProcessor {
public:
    SegmentationPostProcessor(MODEL_TYPE modelType, float rectConfidenceThreshold, float iouThreshold);
    void initBuffers(size_t strideNum) override;
    void PostProcess(void* output, const std::vector<int64_t>& outputNodeDims, std::vector<DL_RESULT>& oResult, float resizeScales, const std::vector<std::string>& classes, void* secondaryOutput = nullptr, const std::vector<int64_t>& secondaryDims = {}) override;
private:
    void greedyNMS(float iouThresh);
    MODEL_TYPE modelType;
    float rectConfidenceThreshold;
    float iouThreshold;
    std::vector<float> m_bestScores;
    std::vector<int> m_bestClassIds;
    std::vector<int> m_classIds;
    std::vector<float> m_confidences;
    std::vector<cv::Rect> m_boxes;
    std::vector<std::vector<float>> m_maskCoeffs;
    std::vector<int> m_nmsIndices;
    std::vector<int> m_sortIndices;
    std::vector<bool> m_suppressed;
};


