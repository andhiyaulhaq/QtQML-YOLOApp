#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

struct LetterboxInfo {
    float scale = 1.0f;
    int padW = 0;
    int padH = 0;
};

struct DetectionResult {

    int        classId;
    float      confidence;
    cv::Rect   box;
    std::vector<cv::Point2f> keyPoints;
    cv::Mat    boxMask;
};
