#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "../domain/TaskType.h"
#include "../domain/DetectionResult.h"

class IPostProcessor {
public:
    virtual ~IPostProcessor() = default;
    virtual void initBuffers(size_t strideNum) = 0;
    virtual void postProcess(void* output, const std::vector<int64_t>& outputNodeDims, std::vector<DetectionResult>& oResult, const LetterboxInfo& info, const std::vector<std::string>& classes, void* secondaryOutput = nullptr, const std::vector<int64_t>& secondaryDims = {}) = 0;
};

class DetectionPostProcessor : public IPostProcessor {
public:
    DetectionPostProcessor(YoloTask::TaskType taskType, float rectConfidenceThreshold, float iouThreshold);
    void initBuffers(size_t strideNum) override;
    void postProcess(void* output, const std::vector<int64_t>& outputNodeDims, std::vector<DetectionResult>& oResult, const LetterboxInfo& info, const std::vector<std::string>& classes, void* secondaryOutput = nullptr, const std::vector<int64_t>& secondaryDims = {}) override;
private:
    void greedyNMS(float iouThresh);
    YoloTask::TaskType m_taskType;
    float m_rectConfidenceThreshold;
    float m_iouThreshold;
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
    PosePostProcessor(YoloTask::TaskType taskType, float rectConfidenceThreshold, float iouThreshold);
    void initBuffers(size_t strideNum) override;
    void postProcess(void* output, const std::vector<int64_t>& outputNodeDims, std::vector<DetectionResult>& oResult, const LetterboxInfo& info, const std::vector<std::string>& classes, void* secondaryOutput = nullptr, const std::vector<int64_t>& secondaryDims = {}) override;
private:
    void greedyNMS(float iouThresh);
    YoloTask::TaskType m_taskType;
    float m_rectConfidenceThreshold;
    float m_iouThreshold;
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
    SegmentationPostProcessor(YoloTask::TaskType taskType, float rectConfidenceThreshold, float iouThreshold);
    void initBuffers(size_t strideNum) override;
    void postProcess(void* output, const std::vector<int64_t>& outputNodeDims, std::vector<DetectionResult>& oResult, const LetterboxInfo& info, const std::vector<std::string>& classes, void* secondaryOutput = nullptr, const std::vector<int64_t>& secondaryDims = {}) override;
private:
    void greedyNMS(float iouThresh);
    YoloTask::TaskType m_taskType;
    float m_rectConfidenceThreshold;
    float m_iouThreshold;
    std::vector<float> m_bestScores;
    std::vector<int> m_bestClassIds;
    std::vector<int> m_classIds;
    std::vector<float> m_confidences;
    std::vector<cv::Rect> m_boxes;
    std::vector<std::vector<float>> m_maskCoeffs;
    std::vector<int> m_nmsIndices;
    std::vector<int> m_sortIndices;
    std::vector<bool> m_suppressed;

    cv::Mat m_protoMat;
    cv::Mat m_maskMat;
    cv::Mat m_maskResized;
};
