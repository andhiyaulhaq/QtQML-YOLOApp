#pragma once

#include <opencv2/opencv.hpp>
#include <memory>

struct CameraFrame {
    std::shared_ptr<cv::Mat> mat;
    qint64 timestamp;
};
