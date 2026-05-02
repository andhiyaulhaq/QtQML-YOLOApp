#pragma once

#include <string>
#include <vector>
#include <thread>
#include <algorithm>
#include "TaskType.h"

struct InferenceConfig {
    std::string modelPath;
    YoloTask::TaskType    taskType    = YoloTask::TaskType::ObjectDetection;
    YoloTask::RuntimeType runtimeType = YoloTask::RuntimeType::OpenVINO;
    std::vector<int> imgSize = {640, 640};
    float confidenceThreshold = 0.4f;
    float iouThreshold        = 0.5f;
    int   keyPointsNum        = 2; // Default for pose estimation if needed
    bool  cudaEnable          = false;
    int   intraOpThreads      = std::max(1u, std::thread::hardware_concurrency() / 2);
    int   interOpThreads      = 1;
};
