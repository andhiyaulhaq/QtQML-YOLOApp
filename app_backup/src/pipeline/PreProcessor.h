#pragma once

#include "YoloTypes.h"
#include <opencv2/opencv.hpp>
#include <vector>

class ImagePreProcessor {
public:
    ImagePreProcessor(MODEL_TYPE modelType, const std::vector<int>& imgSize);

    LetterboxInfo PreProcess(const cv::Mat &iImg, cv::Mat &oImg);
    void PreProcessImageToBlob(const cv::Mat& iImg, float* blob_data);

    float getResizeScales() const { return m_info.scale; }

private:
    MODEL_TYPE modelType;
    std::vector<int> imgSize;
    LetterboxInfo m_info;
};
