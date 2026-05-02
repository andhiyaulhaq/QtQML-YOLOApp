#pragma once

#include <memory>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "../domain/IDetectionModel.h"
#include "PreProcessor.h"
#include "PostProcessor.h"
#include "backends/IInferenceBackend.h"

class YoloPipeline : public IDetectionModel {
public:
    YoloPipeline();
    ~YoloPipeline() override;

    const char* createSession(const InferenceConfig& config) override;
    char* runInference(const cv::Mat& frame,
                       std::vector<DetectionResult>& results,
                       InferenceTiming& timing) override;
    const std::vector<std::string>& classNames() const override { return m_classes; }
    void warmUp() override;

private:
    std::unique_ptr<IInferenceBackend> m_backend;
    std::unique_ptr<ImagePreProcessor> m_preProcessor;
    std::unique_ptr<IPostProcessor> m_postProcessor;

    YoloTask::TaskType m_taskType;
    std::vector<int> m_imgSize;
    std::vector<std::string> m_classes;

    // Optimization: Reusable memory for blob to avoid reallocations
    cv::Mat m_commonBlob; 
    cv::Mat m_letterboxBuffer;
};
