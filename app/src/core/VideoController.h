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
#include <mutex> // Added for std::mutex and std::lock_guard
#include <opencv2/opencv.hpp>
#include <chrono>
#include <memory>
#include <QMediaDevices>
#include <QCameraDevice>

namespace AppConfig {
    static constexpr int FrameWidth = 640;
    static constexpr int FrameHeight = 480;
    static constexpr int ModelWidth = 640;
    static constexpr int ModelHeight = 640;
}

#include "YoloPipeline.h"
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

public:
    enum TaskType {
        TaskObjectDetection = 1,
        TaskPoseEstimation = 2,
        TaskImageSegmentation = 3
    };
    Q_ENUM(TaskType)

    enum RuntimeType {
        RuntimeOpenVINO = 0,
        RuntimeONNXRuntime = 1
    };
    Q_ENUM(RuntimeType)

    Q_PROPERTY(TaskType currentTask READ currentTask WRITE setCurrentTask NOTIFY currentTaskChanged)
    Q_PROPERTY(RuntimeType currentRuntime READ currentRuntime WRITE setCurrentRuntime NOTIFY currentRuntimeChanged)
    Q_PROPERTY(QVideoSink* videoSink READ videoSink WRITE setVideoSink NOTIFY videoSinkChanged)
    Q_PROPERTY(double fps READ fps NOTIFY fpsChanged)
    Q_PROPERTY(QString systemStats READ systemStats NOTIFY systemStatsChanged)
    Q_PROPERTY(QObject* detections READ detections NOTIFY detectionsChanged)
    Q_PROPERTY(double preProcessTime READ preProcessTime NOTIFY timingChanged)
    Q_PROPERTY(double inferenceTime READ inferenceTime NOTIFY timingChanged)
    Q_PROPERTY(double postProcessTime READ postProcessTime NOTIFY timingChanged)
    Q_PROPERTY(double inferenceFps READ inferenceFps NOTIFY inferenceFpsChanged)
    Q_PROPERTY(QVariantList supportedResolutions READ supportedResolutions NOTIFY supportedResolutionsChanged)
    Q_PROPERTY(QSize currentResolution READ currentResolution WRITE setCurrentResolution NOTIFY currentResolutionChanged)

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
    TaskType currentTask() const { return m_currentTask; }
    RuntimeType currentRuntime() const { return m_currentRuntime; }
    QVariantList supportedResolutions() const { return m_supportedResolutions; }
    QSize currentResolution() const { return m_currentResolution; }

signals:
    void currentTaskChanged();
    void currentRuntimeChanged();
    void taskChangedBus(int taskType);
    void runtimeChangedBus(int runtimeType);
    void videoSinkChanged();
    void fpsChanged();
    void inferenceFpsChanged();
    void systemStatsChanged();
    void detectionsChanged();
    void timingChanged();
    void errorOccurred(const QString& title, const QString& message);
    void supportedResolutionsChanged();
    void currentResolutionChanged();
    
    // Signals to workers
    void startWorkers(QVideoSink* sink);
    void stopWorkers();

public slots:
    void setCurrentTask(TaskType task);
    void setCurrentRuntime(RuntimeType runtime);
    void updateFps(double fps);
    void updateSystemStats(const QString &formattedStats);
    void updateDetections(const std::vector<DL_RESULT>& results, const std::vector<std::string>* classNames, const YoloPipeline::InferenceTiming& timing);
    void setCurrentResolution(const QSize& size);

private slots:
    void handleInferenceError(const QString& title, const QString& message);
    void handleModelLoaded(int task, int runtime);
    void refreshResolutions();

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
    
    TaskType m_currentTask = TaskObjectDetection;
    RuntimeType m_currentRuntime = RuntimeOpenVINO;
    TaskType m_lastGoodTask = TaskObjectDetection;
    RuntimeType m_lastGoodRuntime = RuntimeOpenVINO;

    QVariantList m_supportedResolutions;
    QSize m_currentResolution = QSize(640, 480);

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
    void resolutionChanged(QSize size);
    void cleanUp();

public slots:
    void startCapturing(QVideoSink* sink);
    void stopCapturing();
    void setInferenceProcessingFlag(std::atomic<bool>* flag) { m_inferenceProcessingFlag = flag; }
    void updateLatestDetections(std::shared_ptr<std::vector<DL_RESULT>> detections);
    void clearDetections();
    void updateResolution(const QSize& size);

private:
    std::mutex m_detectionsMutex;
    std::shared_ptr<std::vector<DL_RESULT>> m_latestDetections;
    std::atomic<bool> m_running{false};
    std::atomic<bool>* m_inferenceProcessingFlag = nullptr;
    cv::VideoCapture m_capture;
    
    std::mutex m_resolutionMutex;
    QSize m_requestedResolution = QSize(640, 480);
    std::atomic<bool> m_resolutionUpdatePending{false};
    QVideoSink* m_sink = nullptr;
    
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
    void detectionsReady(const std::vector<DL_RESULT>& results, const std::vector<std::string>* classNames, const YoloPipeline::InferenceTiming& timing);
    void latestDetectionsReady(std::shared_ptr<std::vector<DL_RESULT>> results);
    void modelLoaded(int taskType, int runtimeType);
    void errorOccurred(const QString& title, const QString& message);

public slots:
    void startInference(); // Initialize model
    void stopInference();
    void changeModel(int taskType);
    void changeRuntime(int runtimeType);
    void processFrame(std::shared_ptr<cv::Mat> frame);
    std::atomic<bool>* getProcessingFlag() { return &m_isProcessing; }

private:
    std::atomic<bool> m_running{false};
    std::unique_ptr<YoloPipeline> m_pipeline;
    
    int m_currentTaskType = 1;
    int m_currentRuntimeType = 0; // RuntimeOpenVINO

    // Drop frames if inference is too slow
    std::atomic<bool> m_isProcessing{false};
};

Q_DECLARE_METATYPE(std::shared_ptr<cv::Mat>)
Q_DECLARE_METATYPE(std::shared_ptr<std::vector<DL_RESULT>>)

#endif // VIDEOCONTROLLER_H
