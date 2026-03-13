#pragma once

#include "YoloTypes.h"
#include <opencv2/opencv.hpp>
#include <vector>

class ImagePreProcessor {
public:
    ImagePreProcessor(MODEL_TYPE modelType, const std::vector<int>& imgSize);

    char* PreProcess(const cv::Mat &iImg, cv::Mat &oImg);
    void PreProcessImageToBlob(const cv::Mat& iImg, float* blob_data);

    float getResizeScales() const { return resizeScales; }
    void setResizeScales(float s) { resizeScales = s; }

private:
    MODEL_TYPE modelType;
    std::vector<int> imgSize;
    float resizeScales = 1.0f;
};
