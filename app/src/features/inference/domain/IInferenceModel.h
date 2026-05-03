#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "InferenceResult.h"
#include "InferenceConfig.h"
#include "InferenceTiming.h"

class IInferenceModel {
public:
    virtual ~IInferenceModel() = default;
    virtual const char* createSession(const InferenceConfig& config) = 0;
    virtual char* runInference(const cv::Mat& frame,
                               std::vector<InferenceResult>& results,
                               InferenceTiming& timing) = 0;
    virtual const std::vector<std::string>& classNames() const = 0;
    virtual void warmUp() = 0;
};
