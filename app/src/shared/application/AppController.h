#pragma once

#include <QObject>
#include <QQmlApplicationEngine>
#include <QThread>

// Forward declarations
class MonitoringController;
class SystemMonitorWorker;
class ISystemMonitor;

class DetectionController;
class InferenceWorker;
class IDetectionModel;

class YoloCameraController;
class VideoFileController;
class CaptureWorker;
class ICaptureSource;

class AppController : public QObject {
    Q_OBJECT

public:
    explicit AppController(QQmlApplicationEngine *engine, QObject *parent = nullptr);
    ~AppController() override;

    void initialize();

private:
    QQmlApplicationEngine *m_engine;

    // Monitoring Feature
    ISystemMonitor *m_systemMonitorImpl;
    SystemMonitorWorker *m_monitoringWorker;
    MonitoringController *m_monitoringController;
    QThread m_monitoringThread;

    // Detection Feature
    IDetectionModel *m_detectionModelImpl;
    InferenceWorker *m_inferenceWorker;
    DetectionController *m_detectionController;
    QThread m_inferenceThread;

    // Capture/Camera Feature
    ICaptureSource *m_captureSourceImpl;
    CaptureWorker *m_captureWorker;
    YoloCameraController *m_cameraController;
    VideoFileController *m_videoFileController;
    QThread m_cameraThread;

    void setupMonitoring();
    void setupDetection();
    void setupCamera();
    void wireEverything();
};
