#pragma once

#include "ICaptureSource.h"
#include <opencv2/opencv.hpp>

class OpenCVImageFileSource : public ICaptureSource {
public:
    OpenCVImageFileSource() = default;
    ~OpenCVImageFileSource() override;

    bool open(const SourceConfig& config) override;
    void close() override;
    bool readFrame(cv::Mat& outFrame) override;
    QSize currentResolution() const override;

private:
    cv::Mat m_image;
    QSize m_resolution;
};
