#pragma once

#include "../domain/ICaptureSource.h"
#include <opencv2/opencv.hpp>
#include <QString>

class OpenCVVideoFileSource : public ICaptureSource {
public:
    OpenCVVideoFileSource() = default;
    ~OpenCVVideoFileSource() override;

    bool open(const SourceConfig& config) override;
    void close() override;
    bool readFrame(cv::Mat& outFrame) override;
    bool skipFrame();
    QSize currentResolution() const override;

    double nativeFps() const;

private:
    cv::VideoCapture m_capture;
    QSize m_currentResolution;
    double m_nativeFps = 30.0;
    bool m_loop = true;
    QString m_filePath;
};
