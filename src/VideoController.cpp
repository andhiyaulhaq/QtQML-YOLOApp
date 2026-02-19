#include "VideoController.h"
#include <QDebug>
#include <QThread>
#include <fstream>
#include <algorithm>

// =========================================================
// CAPTURE WORKER
// =========================================================

void CaptureWorker::startCapturing(QVideoSink* sink) {
    if (m_running) return;
    m_running = true;

    // Open Camera (DSHOW for Windows usually faster)
    m_capture.open(0, cv::CAP_DSHOW);
    if (!m_capture.isOpened()) {
        m_capture.open(0);
    }

    if (!m_capture.isOpened()) {
        // qWarning() << "Could not open camera";
        m_running = false;
        return;
    }

    // Optimization Settings
    m_capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    m_capture.set(cv::CAP_PROP_FRAME_WIDTH, AppConfig::FrameWidth);
    m_capture.set(cv::CAP_PROP_FRAME_HEIGHT, AppConfig::FrameHeight);
    m_capture.set(cv::CAP_PROP_FPS, 30);

    // cv::Mat rawFrame; // Replaced by pool
    int frames = 0;
    auto startTime = std::chrono::high_resolution_clock::now();

    while (m_running) {
        cv::Mat& currentFrame = m_framePool[m_poolIndex];
        
        if (!m_capture.read(currentFrame) || currentFrame.empty()) {
            QThread::msleep(10);
            continue;
        }

        // 1. Send to UI (Zero-copy optimization)
        if (sink) {
            QVideoFrameFormat format(QSize(currentFrame.cols, currentFrame.rows),
                                   QVideoFrameFormat::Format_RGBA8888);
            QVideoFrame frame(format);
            
            if (frame.map(QVideoFrame::WriteOnly)) {
                cv::Mat wrapper(currentFrame.rows, currentFrame.cols, CV_8UC4, 
                              frame.bits(0), frame.bytesPerLine(0));
                cv::cvtColor(currentFrame, wrapper, cv::COLOR_BGR2RGBA);
                frame.unmap();
                sink->setVideoFrame(frame);
            }
        }

        // 2. Send to Inference (Optimization: Ring Buffer instead of clone)
        // We emit the Mat. Qt copies the header, increasing the refcounter to the underlying data.
        // As long as we don't overwrite this specific m_framePool index before inference is done, we are safe.
        // With 3 buffers at 30fps, we have 100ms before overwrite. If inference < 100ms, correct.
        emit frameReady(currentFrame);
        
        // Advance pool index
        m_poolIndex = (m_poolIndex + 1) % 3;

        // 3. FPS Calculation
        frames++;
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();
        if (duration >= 1000) {
            double fps = frames * 1000.0 / duration;
            emit fpsUpdated(fps);
            frames = 0;
            startTime = now;
        }
    }
    m_capture.release();
    emit cleanUp();
}

void CaptureWorker::stopCapturing() {
    m_running = false;
}

// =========================================================
// INFERENCE WORKER
// =========================================================

InferenceWorker::InferenceWorker() {}

InferenceWorker::~InferenceWorker() {
    stopInference();
    if (m_yolo) delete m_yolo;
}

void InferenceWorker::startInference() {
    m_yolo = new YOLO_V8;
    
    // Load Classes
    std::ifstream file("inference/classes.txt");
    std::string line;
    while (std::getline(file, line)) {
        m_yolo->classes.push_back(line);
        m_classNames.push_back(line);
    }
    file.close();

    DL_INIT_PARAM params;
    params.modelPath = "inference/yolov8n.onnx";
    params.modelType = YOLO_DETECT_V8;
    params.imgSize = {AppConfig::ModelWidth, AppConfig::ModelHeight};
    params.cudaEnable = false; // CPU
    // Optimization: Cap threads to 4 for YOLOv8n (small model). 
    // More threads often increase overhead without performance gain for 'nano' models.
    unsigned int threads = std::thread::hardware_concurrency() / 2;
    params.intraOpNumThreads = std::max(1u, std::min(4u, threads)); 
    params.interOpNumThreads = 1;
    m_yolo->CreateSession(params);
    
    m_running = true;
}

void InferenceWorker::stopInference() {
    m_running = false;
}

void InferenceWorker::processFrame(const cv::Mat& frame) {
    if (!m_running || !m_yolo) return;

    // Drop frame if already processing
    bool expected = false;
    if (!m_isProcessing.compare_exchange_strong(expected, true)) {
        return; 
    }

    // Run Inference
    std::vector<DL_RESULT> results;
    YOLO_V8::InferenceTiming timing;
    // We can use the input const reference directly if RunSession accepts it.
    // If not, we might need a const_cast or better, fix RunSession to take const cv::Mat&
    // For now, let's assume we will fix RunSession.
    m_yolo->RunSession(frame, results, timing);

    // Emit results
    emit detectionsReady(results, m_classNames, timing);

    m_isProcessing = false;
}

// =========================================================
// VIDEO CONTROLLER
// =========================================================

VideoController::VideoController(QObject *parent) : QObject(parent) {
    qRegisterMetaType<cv::Mat>("cv::Mat");
    
    // Initialize Model
    m_detections = new DetectionListModel(this);
    // 1. Create Workers
    m_captureWorker = new CaptureWorker();
    m_inferenceWorker = new InferenceWorker();

    m_captureWorker->moveToThread(&m_captureThread);
    m_inferenceWorker->moveToThread(&m_inferenceThread);

    // 2. Wiring

    // Main -> Workers
    connect(this, &VideoController::startWorkers, m_captureWorker, &CaptureWorker::startCapturing, Qt::QueuedConnection);
    connect(this, &VideoController::stopWorkers, m_captureWorker, &CaptureWorker::stopCapturing, Qt::DirectConnection); 
    connect(this, &VideoController::stopWorkers, m_inferenceWorker, &InferenceWorker::stopInference, Qt::DirectConnection); 
    connect(&m_inferenceThread, &QThread::started, m_inferenceWorker, &InferenceWorker::startInference);
    
    // Capture -> Inference
    connect(m_captureWorker, &CaptureWorker::frameReady, m_inferenceWorker, &InferenceWorker::processFrame, Qt::QueuedConnection);

    // Workers -> Main
    connect(m_captureWorker, &CaptureWorker::fpsUpdated, this, &VideoController::updateFps);
    connect(m_inferenceWorker, &InferenceWorker::detectionsReady, this, &VideoController::updateDetections);

    // System Monitor
    m_systemMonitor = new SystemMonitor(nullptr); // No parent to allow moveToThread
    m_systemMonitor->moveToThread(&m_systemThread);
    connect(&m_systemThread, &QThread::finished, m_systemMonitor, &QObject::deleteLater);
    connect(m_systemMonitor, &SystemMonitor::resourceUsageUpdated, this, &VideoController::updateSystemStats);
    
    // Wire start/stop
    connect(this, &VideoController::startWorkers, m_systemMonitor, &SystemMonitor::startMonitoring, Qt::QueuedConnection);
    connect(this, &VideoController::stopWorkers, m_systemMonitor, &SystemMonitor::stopMonitoring, Qt::QueuedConnection);

    // Start Threads
    // Start Threads (Deferred to prevent startup hang)
    QTimer::singleShot(500, this, [this](){
        m_inferenceThread.start(QThread::HighPriority);
        m_captureThread.start();
        m_systemThread.start(QThread::LowPriority);
    });
    
    m_lastInferenceTime = std::chrono::steady_clock::now();
}

VideoController::~VideoController() {
    emit stopWorkers();
    
    // Signal all threads to quit
    m_captureThread.quit();
    m_inferenceThread.quit();
    m_systemThread.quit();
    
    // Wait for all to finish (parallelized wait)
    m_captureThread.wait();
    m_inferenceThread.wait();
    m_systemThread.wait();

    delete m_captureWorker;
    delete m_inferenceWorker;
    // SystemMonitor deleted by deleteLater on thread finish
}

void VideoController::setVideoSink(QVideoSink* sink) {
    if (m_sink == sink) return;
    m_sink = sink;
    emit videoSinkChanged();

    if (m_sink) {
        emit startWorkers(m_sink);
        // m_systemMonitor->startMonitoring(); // Removed direct call
    } else {
        emit stopWorkers();
        // m_systemMonitor->stopMonitoring(); // Removed direct call
    }
}

void VideoController::updateFps(double fps) {
    if (std::abs(m_fps - fps) > 0.1) {
        m_fps = fps;
        emit fpsChanged();
    }
}

void VideoController::updateSystemStats(const QString &cpu, const QString &sysMem, const QString &procMem) {
    QString stats = QString("CPU: %1 | RAM: %2").arg(cpu, procMem);
    if (m_systemStats != stats) {
        m_systemStats = stats;
        emit systemStatsChanged();
    }
}

void VideoController::updateDetections(const std::vector<DL_RESULT>& results, const std::vector<std::string>& classNames, const YOLO_V8::InferenceTiming& timing) {
    // Update the model directly
    if (m_detections) {
        m_detections->updateDetections(results, classNames);
        emit detectionsChanged(); // Notify that the model object itself 'changed' (or just keeping signal for compatibility, mostly not needed if model internal signals fire)
        // Actually, for QAbstractListModel, the model emits rowsInserted/etc. 
        // emit detectionsChanged() is only needed if the pointer m_detections changed, which it doesn't.
        // But some QML bindings might rely on it if they bind to 'detections'.
        // Let's keep it but it might be redundant.
    }

    // Update timing
    if (std::abs(m_preProcessTime - timing.preProcessTime) > 0.001 ||
        std::abs(m_inferenceTime - timing.inferenceTime) > 0.001 ||
        std::abs(m_postProcessTime - timing.postProcessTime) > 0.001) {
        
        m_preProcessTime = timing.preProcessTime;
        m_inferenceTime = timing.inferenceTime;
        m_postProcessTime = timing.postProcessTime;
        emit timingChanged();
    }

    // Calculate Inference FPS
    auto now = std::chrono::steady_clock::now();
    double diff = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_lastInferenceTime).count();
    
    if (diff > 0) {
        double currentFps = 1000.0 / diff;
        // Simple smoothing
        if (m_inferenceFps == 0) m_inferenceFps = currentFps;
        else m_inferenceFps = m_inferenceFps * 0.9 + currentFps * 0.1;
        
        emit inferenceFpsChanged();
    }
    m_lastInferenceTime = now;
}
