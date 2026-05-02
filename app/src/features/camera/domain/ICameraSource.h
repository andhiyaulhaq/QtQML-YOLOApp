#pragma once

#include <opencv2/opencv.hpp>
#include <QSize>
#include "CameraConfig.h"

class ICameraSource {
public:
    virtual ~ICameraSource() = default;
    virtual bool open(const CameraConfig& config) = 0;
    virtual void close() = 0;
    virtual bool readFrame(cv::Mat& outFrame) = 0;
    virtual QSize currentResolution() const = 0;
};
