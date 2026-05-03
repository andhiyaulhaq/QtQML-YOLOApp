#pragma once

#include "../domain/ICaptureSource.h"
#include <string>
#include <mutex>
#include <atomic>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
#include <libavutil/display.h>
}

class FFmpegVideoSource : public ICaptureSource {
public:
    FFmpegVideoSource();
    ~FFmpegVideoSource() override;

    bool open(const SourceConfig& config) override;
    void close() override;
    bool readFrame(cv::Mat& outFrame) override;
    QSize currentResolution() const override;
    int64_t frameCount() const override;
    int64_t currentFrameIndex() const override;
    double nativeFps() const override;
    bool seekToFrame(int64_t frameIndex) override;
    bool skipFrame() override;

private:
    bool initFFmpeg(const std::string& path);
    void cleanup();
    bool readFrameInternal(cv::Mat& outFrame);

    AVFormatContext* m_formatCtx = nullptr;
    AVCodecContext* m_codecCtx = nullptr;
    int m_videoStreamIndex = -1;
    AVFrame* m_frame = nullptr;
    AVFrame* m_frameRGB = nullptr;
    AVPacket* m_packet = nullptr;
    SwsContext* m_swsCtx = nullptr;
    uint8_t* m_buffer = nullptr;

    int64_t m_currentFrameIndex = 0;
    int64_t m_totalFrames = 0;
    double m_fps = 0.0;
    QSize m_resolution;
    int m_rotation = 0;

    mutable std::mutex m_mutex;
};
