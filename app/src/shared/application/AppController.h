#pragma once

#include <QObject>
#include <QQmlApplicationEngine>
#include <QThread>

// Forward declarations
class MonitoringController;
class SystemMonitorWorker;
class ISystemMonitor;

class InferenceController;
class InferenceWorker;
class IInferenceModel;

class YoloCameraController;
class VideoFileController;
class ImageFileController;
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

    // Inference Feature
    IInferenceModel *m_inferenceModelImpl;
    InferenceWorker *m_inferenceWorker;
    InferenceController *m_inferenceController;
    QThread m_inferenceThread;

    // Capture/Camera Feature
    ICaptureSource *m_captureSourceImpl;
    CaptureWorker *m_captureWorker;
    YoloCameraController *m_cameraController;
    VideoFileController *m_videoFileController;
    ImageFileController *m_imageFileController;
    QThread m_cameraThread;

    void setupMonitoring();
    void setupInference();
    void setupCamera();
    void wireEverything();
};
