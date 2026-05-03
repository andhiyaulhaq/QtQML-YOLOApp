#pragma once

#include <QObject>
#include <QVideoSink>
#include <QVideoFrame>
#include <QMutex>
#include <QSize>
#include <atomic>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include "../domain/ICaptureSource.h"
#include "../../detection/domain/DetectionResult.h"

class CaptureWorker : public QObject {
    Q_OBJECT

public:
    explicit CaptureWorker(ICaptureSource *source, QObject *parent = nullptr);
    ~CaptureWorker() override;

    void setInferenceProcessingFlag(std::atomic<bool>* flag) { m_inferenceProcessingFlag = flag; }

signals:
    void frameReady(std::shared_ptr<cv::Mat> frame); 
    void fpsUpdated(double fps);
    void resolutionChanged(QSize size);
    void metadataUpdated(double fps, int64_t totalFrames);
    void progressUpdated(int64_t frame);
    void cleanUp();

public slots:
    void startCapturing(QVideoSink* sink);
    void stopCapturing();
    void updateLatestDetections(std::shared_ptr<std::vector<DetectionResult>> detections, const QSize& frameSize);
    void clearDetections();
    void updateResolution(const QSize& size);
    void setSource(ICaptureSource* source, const SourceConfig& config);
    void requestSeek(int64_t frame);
    void forceReinference() { m_needsStaticInference = true; }

private:
    std::atomic<bool> m_needsStaticInference{true};
    ICaptureSource *m_source;
    std::mutex m_sourceMutex;

    std::atomic<bool> m_running{false};
    std::atomic<bool>* m_inferenceProcessingFlag = nullptr;
    QVideoSink* m_sink = nullptr;

    std::mutex m_detectionsMutex;
    std::shared_ptr<std::vector<DetectionResult>> m_latestDetections;

    std::mutex m_configMutex;
    SourceConfig m_requestedConfig;
    std::atomic<bool> m_configUpdatePending{false};

    cv::Mat m_framePool[3]; 
    int m_poolIndex = 0;
    
    QVideoFrame m_reusableFrames[2];
    int m_reusableFrameIndex = 0;

    // Video Sync
    std::chrono::time_point<std::chrono::high_resolution_clock> m_videoStartTime;
    int64_t m_videoFramesRead = 0;
    bool m_isFirstFrame = true;

    bool openSource(const SourceConfig& config);
};
