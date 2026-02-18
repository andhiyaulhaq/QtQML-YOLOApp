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

#include "inference.h"
#include "SystemMonitor.h"

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
    Q_PROPERTY(QVariantList detections READ detections NOTIFY detectionsChanged)
    Q_PROPERTY(double preProcessTime READ preProcessTime NOTIFY timingChanged)
    Q_PROPERTY(double inferenceTime READ inferenceTime NOTIFY timingChanged)
    Q_PROPERTY(double postProcessTime READ postProcessTime NOTIFY timingChanged)

public:
    explicit VideoController(QObject *parent = nullptr);
    ~VideoController();

    QVideoSink* videoSink() const { return m_sink; }
    void setVideoSink(QVideoSink* sink);

    double fps() const { return m_fps; }
    QString systemStats() const { return m_systemStats; }
    QVariantList detections() const { return m_detections; }
    
    double preProcessTime() const { return m_preProcessTime; }
    double inferenceTime() const { return m_inferenceTime; }
    double postProcessTime() const { return m_postProcessTime; }

signals:
    void videoSinkChanged();
    void fpsChanged();
    void systemStatsChanged();
    void detectionsChanged();
    void timingChanged();
    
    // Signals to workers
    void startWorkers(QVideoSink* sink);
    void stopWorkers();

public slots:
    void updateFps(double fps);
    void updateSystemStats(const QString &cpu, const QString &sysMem, const QString &procMem);
    void updateDetections(const std::vector<DL_RESULT>& results, const std::vector<std::string>& classNames, const YOLO_V8::InferenceTiming& timing);

private:
    QVideoSink* m_sink = nullptr;
    double m_fps = 0.0;
    QString m_systemStats;
    QVariantList m_detections;
    
    double m_preProcessTime = 0.0;
    double m_inferenceTime = 0.0;
    double m_postProcessTime = 0.0;

    // Workers and Threads
    CaptureWorker* m_captureWorker = nullptr;
    InferenceWorker* m_inferenceWorker = nullptr;
    QThread m_captureThread;
    QThread m_inferenceThread;

    SystemMonitor* m_systemMonitor = nullptr;
};

// =========================================================
// WORKERS (Background Threads)
// =========================================================

class CaptureWorker : public QObject {
    Q_OBJECT
public:
    CaptureWorker() = default;

signals:
    void frameReady(const cv::Mat& frame); // Send frame to Inference Worker
    void fpsUpdated(double fps);
    void cleanUp();

public slots:
    void startCapturing(QVideoSink* sink);
    void stopCapturing();

private:
    std::atomic<bool> m_running{false};
    cv::VideoCapture m_capture;
};

class InferenceWorker : public QObject {
    Q_OBJECT
public:
    InferenceWorker();
    ~InferenceWorker();

signals:
    void detectionsReady(const std::vector<DL_RESULT>& results, const std::vector<std::string>& classNames, const YOLO_V8::InferenceTiming& timing);

public slots:
    void startInference(); // Initialize model
    void stopInference();
    void processFrame(const cv::Mat& frame);

private:
    std::atomic<bool> m_running{false};
    YOLO_V8* m_yolo = nullptr;
    std::vector<std::string> m_classNames;
    
    // Drop frames if inference is too slow
    std::atomic<bool> m_isProcessing{false};
};

#endif // VIDEOCONTROLLER_H
