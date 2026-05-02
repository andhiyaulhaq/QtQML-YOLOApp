#pragma once

#include "../domain/ICameraSource.h"
#include <opencv2/opencv.hpp>

class OpenCVCameraSource : public ICameraSource {
public:
    OpenCVCameraSource();
    ~OpenCVCameraSource() override;

    bool open(const CameraConfig& config) override;
    void close() override;
    bool readFrame(cv::Mat& outFrame) override;
    QSize currentResolution() const override;

private:
    cv::VideoCapture m_capture;
    QSize m_currentResolution;
};
