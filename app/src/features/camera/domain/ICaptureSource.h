#pragma once

#include <opencv2/opencv.hpp>
#include <QSize>
#include "SourceConfig.h"

class ICaptureSource {
public:
    virtual ~ICaptureSource() = default;

    virtual bool open(const SourceConfig& config) = 0;
    virtual void close() = 0;
    virtual bool readFrame(cv::Mat& outFrame) = 0;
    virtual QSize currentResolution() const = 0;
    virtual int64_t frameCount() const { return -1; }
    virtual int64_t currentFrameIndex() const { return -1; }
    virtual bool seekToFrame(int64_t frameIndex) { Q_UNUSED(frameIndex); return false; }
};
