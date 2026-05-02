#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "../domain/TaskType.h"
#include "../domain/DetectionResult.h"

class ImagePreProcessor {
public:
    ImagePreProcessor(YoloTask::TaskType taskType, const std::vector<int>& imgSize);

    LetterboxInfo preProcess(const cv::Mat &iImg, cv::Mat &oImg);
    void preProcessImageToBlob(const cv::Mat& iImg, float* blob_data);

    float getResizeScales() const { return m_info.scale; }

private:
    YoloTask::TaskType m_taskType;
    std::vector<int> m_imgSize;
    LetterboxInfo m_info;
};
