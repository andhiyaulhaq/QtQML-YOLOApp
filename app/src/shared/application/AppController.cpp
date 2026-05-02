#include "AppController.h"
#include <QQmlContext>
#include <QTimer>

// Monitoring
#include "../../features/monitoring/infrastructure/WindowsSystemMonitor.h"
#include "../../features/monitoring/application/SystemMonitorWorker.h"
#include "../../features/monitoring/application/MonitoringController.h"

// Detection
#include "../../features/detection/infrastructure/YoloPipeline.h"
#include "../../features/detection/application/InferenceWorker.h"
#include "../../features/detection/application/DetectionController.h"

// Capture/Camera
#include "../../features/camera/infrastructure/OpenCVCameraSource.h"
#include "../../features/camera/application/CaptureWorker.h"
#include "../../features/camera/application/YoloCameraController.h"
#include "../../features/camera/application/VideoFileController.h"

AppController::AppController(QQmlApplicationEngine *engine, QObject *parent)
    : QObject(parent)
    , m_engine(engine)
{
}

AppController::~AppController()
{
    if (m_captureWorker) m_captureWorker->stopCapturing();
    if (m_inferenceWorker) m_inferenceWorker->stopInference();
    if (m_monitoringWorker) {
        QMetaObject::invokeMethod(m_monitoringWorker, "stop", Qt::BlockingQueuedConnection);
    }

    m_cameraThread.quit();
    m_inferenceThread.quit();
    m_monitoringThread.quit();

    m_cameraThread.wait();
    m_inferenceThread.wait();
    m_monitoringThread.wait();
}

void AppController::initialize()
{
    setupMonitoring();
    setupDetection();
    setupCamera();
    wireEverything();

    m_engine->rootContext()->setContextProperty("monitoring", m_monitoringController);
    m_engine->rootContext()->setContextProperty("detection", m_detectionController);
    m_engine->rootContext()->setContextProperty("camera", m_cameraController);
    m_engine->rootContext()->setContextProperty("videoFile", m_videoFileController);

    m_monitoringThread.start(QThread::LowPriority);
    m_inferenceThread.start(QThread::HighPriority);
    m_cameraThread.start();

    // Start workers
    QMetaObject::invokeMethod(m_monitoringWorker, "start", Qt::QueuedConnection);
}

void AppController::setupMonitoring()
{
    m_systemMonitorImpl = new WindowsSystemMonitor();
    m_monitoringWorker = new SystemMonitorWorker(m_systemMonitorImpl);
    m_monitoringController = new MonitoringController(m_monitoringWorker, this);

    m_monitoringWorker->moveToThread(&m_monitoringThread);
    connect(&m_monitoringThread, &QThread::finished, m_monitoringWorker, &QObject::deleteLater);
}

void AppController::setupDetection()
{
    m_detectionModelImpl = new YoloPipeline();
    m_inferenceWorker = new InferenceWorker(m_detectionModelImpl);
    m_detectionController = new DetectionController(m_inferenceWorker, this);

    m_inferenceWorker->moveToThread(&m_inferenceThread);
    connect(&m_inferenceThread, &QThread::finished, m_inferenceWorker, &QObject::deleteLater);
}

void AppController::setupCamera()
{
    m_captureSourceImpl = new OpenCVCameraSource();
    m_captureWorker = new CaptureWorker(m_captureSourceImpl);
    m_cameraController = new YoloCameraController(m_captureWorker, this);
    m_videoFileController = new VideoFileController(this);

    m_captureWorker->moveToThread(&m_cameraThread);
    connect(&m_cameraThread, &QThread::finished, m_captureWorker, &QObject::deleteLater);
}

void AppController::wireEverything()
{
    // Monitoring
    connect(m_monitoringWorker, &SystemMonitorWorker::statsUpdated, m_monitoringController, &MonitoringController::updateStats);

    // Detection
    connect(m_inferenceWorker, &InferenceWorker::detectionsReady, m_detectionController, &DetectionController::updateDetections);
    connect(m_detectionController, &DetectionController::requestModelChange, m_inferenceWorker, &InferenceWorker::startInference);

    // Capture (Common)
    connect(m_cameraController, &YoloCameraController::startCapture, m_captureWorker, &CaptureWorker::startCapturing);
    connect(m_cameraController, &YoloCameraController::stopCapture, m_captureWorker, &CaptureWorker::stopCapturing);
    connect(m_captureWorker, &CaptureWorker::fpsUpdated, m_cameraController, &YoloCameraController::updateFps);
    connect(m_captureWorker, &CaptureWorker::resolutionChanged, m_cameraController, &YoloCameraController::handleResolutionChanged);

    // Source Ready Requests
    connect(m_cameraController, &YoloCameraController::sourceReadyRequested, m_captureWorker, &CaptureWorker::setSource);
    connect(m_videoFileController, &VideoFileController::sourceReadyRequested, m_captureWorker, &CaptureWorker::setSource);

    // Cross-Feature
    connect(m_captureWorker, &CaptureWorker::frameReady, m_inferenceWorker, &InferenceWorker::processFrame);
    connect(m_inferenceWorker, &InferenceWorker::latestDetectionsReady, m_captureWorker, &CaptureWorker::updateLatestDetections, Qt::DirectConnection);
    m_captureWorker->setInferenceProcessingFlag(m_inferenceWorker->getProcessingFlag());

    // Initial Model Load
    QTimer::singleShot(500, [this](){
        m_detectionController->setCurrentRuntime(YoloTask::RuntimeType::OpenVINO);
        m_detectionController->setCurrentTask(YoloTask::TaskType::ObjectDetection);
        
        // Default to live camera on start
        m_cameraController->activate();
    });
}
