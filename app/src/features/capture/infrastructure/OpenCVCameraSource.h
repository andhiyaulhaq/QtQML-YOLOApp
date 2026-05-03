#pragma once

#include "../domain/ICaptureSource.h"
#include <opencv2/opencv.hpp>

class OpenCVCameraSource : public ICaptureSource {
public:
    OpenCVCameraSource();
    ~OpenCVCameraSource() override;

    bool open(const SourceConfig& config) override;
    void close() override;
    bool readFrame(cv::Mat& outFrame) override;
    QSize currentResolution() const override;

private:
    cv::VideoCapture m_capture;
    QSize m_currentResolution;
};
