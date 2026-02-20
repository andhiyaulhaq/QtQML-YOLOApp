#ifndef VIDEOCONTROLLER_H
#define VIDEOCONTROLLER_H

#include <QQmlEngine>
#include <QObject>
#include <QVideoSink>
#include <QVideoFrame>
#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <QVariantList>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <memory>

namespace AppConfig {
    static constexpr int FrameWidth = 640;
    static constexpr int FrameHeight = 480;
    static constexpr int ModelWidth = 640;
    static constexpr int ModelHeight = 640;
}

#include "inference.h"
#include "SystemMonitor.h"
#include "DetectionStruct.h"
#include "DetectionListModel.h"

// Forward declarations
class CaptureWorker;
class InferenceWorker;

// =========================================================
// CONTROLLER (Main UI Thread)
// =========================================================
class VideoController : public QObject
{
    Q_OBJECT
    QML_ELEMENT
    Q_PROPERTY(QVideoSink* videoSink READ videoSink WRITE setVideoSink NOTIFY videoSinkChanged)
    Q_PROPERTY(double fps READ fps NOTIFY fpsChanged)
    Q_PROPERTY(QString systemStats READ systemStats NOTIFY systemStatsChanged)
    Q_PROPERTY(QObject* detections READ detections NOTIFY detectionsChanged)
    Q_PROPERTY(double preProcessTime READ preProcessTime NOTIFY timingChanged)
    Q_PROPERTY(double inferenceTime READ inferenceTime NOTIFY timingChanged)
    Q_PROPERTY(double postProcessTime READ postProcessTime NOTIFY timingChanged)
    Q_PROPERTY(double inferenceFps READ inferenceFps NOTIFY inferenceFpsChanged)

public:
    explicit VideoController(QObject *parent = nullptr);
    ~VideoController();

    QVideoSink* videoSink() const { return m_sink; }
    void setVideoSink(QVideoSink* sink);

    double fps() const { return m_fps; }
    QString systemStats() const { return m_systemStats; }
    QObject* detections() const { return m_detections; }
    
    double preProcessTime() const { return m_preProcessTime; }
    double inferenceTime() const { return m_inferenceTime; }
    double postProcessTime() const { return m_postProcessTime; }
    double inferenceFps() const { return m_inferenceFps; }

signals:
    void videoSinkChanged();
    void fpsChanged();
    void inferenceFpsChanged();
    void systemStatsChanged();
    void detectionsChanged();
    void timingChanged();
    
    // Signals to workers
    void startWorkers(QVideoSink* sink);
    void stopWorkers();

public slots:
    void updateFps(double fps);
    void updateSystemStats(const QString &formattedStats);
    void updateDetections(const std::vector<DL_RESULT>& results, const YOLO_V8::InferenceTiming& timing);

private:
    QVideoSink* m_sink = nullptr;
    double m_fps = 0.0;
    QString m_systemStats;
    DetectionListModel* m_detections = nullptr;
    
    double m_preProcessTime = 0.0;
    double m_inferenceTime = 0.0;
    double m_postProcessTime = 0.0;
    double m_inferenceFps = 0.0;
    std::chrono::time_point<std::chrono::steady_clock> m_lastInferenceTime;
    std::vector<std::string> m_classNames;

    // Workers and Threads
    CaptureWorker* m_captureWorker = nullptr;
    InferenceWorker* m_inferenceWorker = nullptr;
    QThread m_captureThread;
    QThread m_inferenceThread;

    SystemMonitor* m_systemMonitor = nullptr;
    QThread m_systemThread;
};

// =========================================================
// WORKERS (Background Threads)
// =========================================================

class CaptureWorker : public QObject {
    Q_OBJECT
public:
    CaptureWorker() = default;

signals:
    void frameReady(std::shared_ptr<cv::Mat> frame); // Send frame to Inference Worker
    void fpsUpdated(double fps);
    void cleanUp();

public slots:
    void startCapturing(QVideoSink* sink);
    void stopCapturing();
    void setInferenceProcessingFlag(std::atomic<bool>* flag) { m_inferenceProcessingFlag = flag; }

private:
    std::atomic<bool> m_running{false};
    std::atomic<bool>* m_inferenceProcessingFlag = nullptr;
    cv::VideoCapture m_capture;
    
    // Multi-buffer logic to avoid cloning
    cv::Mat m_framePool[3]; 
    int m_poolIndex = 0;
    
    // UI double-buffering
    QVideoFrame m_reusableFrames[2];
    int m_reusableFrameIndex = 0;
};

class InferenceWorker : public QObject {
    Q_OBJECT
public:
    InferenceWorker();
    ~InferenceWorker();

signals:
    void detectionsReady(const std::vector<DL_RESULT>& results, const YOLO_V8::InferenceTiming& timing);

public slots:
    void startInference(); // Initialize model
    void stopInference();
    void processFrame(std::shared_ptr<cv::Mat> frame);
    std::atomic<bool>* getProcessingFlag() { return &m_isProcessing; }

private:
    std::atomic<bool> m_running{false};
    std::unique_ptr<YOLO_V8> m_yolo;
    
    // Drop frames if inference is too slow
    std::atomic<bool> m_isProcessing{false};
};

Q_DECLARE_METATYPE(std::shared_ptr<cv::Mat>)

#endif // VIDEOCONTROLLER_H
