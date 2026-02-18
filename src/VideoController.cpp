#include "VideoController.h"
#include <QDebug>
#include <QThread>
#include <fstream>

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
    m_capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    m_capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    m_capture.set(cv::CAP_PROP_FPS, 30);

    cv::Mat rawFrame;
    int frames = 0;
    auto startTime = std::chrono::high_resolution_clock::now();

    while (m_running) {
        if (!m_capture.read(rawFrame) || rawFrame.empty()) {
            QThread::msleep(10);
            continue;
        }

        // 1. Send to UI (Zero-copy optimization)
        if (sink) {
            QVideoFrameFormat format(QSize(rawFrame.cols, rawFrame.rows),
                                   QVideoFrameFormat::Format_RGBA8888);
            QVideoFrame frame(format);
            
            if (frame.map(QVideoFrame::WriteOnly)) {
                cv::Mat wrapper(rawFrame.rows, rawFrame.cols, CV_8UC4, 
                              frame.bits(0), frame.bytesPerLine(0));
                cv::cvtColor(rawFrame, wrapper, cv::COLOR_BGR2RGBA);
                frame.unmap();
                sink->setVideoFrame(frame);
            }
        }

        // 2. Send to Inference (Signal blocks if connected via QueuedConnection, but we want it async)
        // We emit a copy or ensure the receiver handles it quickly. 
        // Ideally we'd use a shared pointer or a circular buffer, but for now copying the Mat is safer to avoid race conditions 
        // given OpenCV Mats differ in thread safety.
        // HOWEVER, to avoid deep copy every frame if inference is slow, the InferenceWorker has a "processing" flag.
        emit frameReady(rawFrame.clone()); 

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
    params.imgSize = {640, 640};
    params.cudaEnable = false; // CPU
    params.intraOpNumThreads = std::thread::hardware_concurrency(); 
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
    // 1. Create Workers
    m_captureWorker = new CaptureWorker();
    m_inferenceWorker = new InferenceWorker();

    m_captureWorker->moveToThread(&m_captureThread);
    m_inferenceWorker->moveToThread(&m_inferenceThread);

    // 2. Wiring

    // Main -> Workers
    connect(this, &VideoController::startWorkers, m_captureWorker, &CaptureWorker::startCapturing, Qt::QueuedConnection);
    connect(this, &VideoController::stopWorkers, m_captureWorker, &CaptureWorker::stopCapturing, Qt::DirectConnection); 
    connect(&m_inferenceThread, &QThread::started, m_inferenceWorker, &InferenceWorker::startInference);
    
    // Capture -> Inference
    connect(m_captureWorker, &CaptureWorker::frameReady, m_inferenceWorker, &InferenceWorker::processFrame, Qt::QueuedConnection);

    // Workers -> Main
    connect(m_captureWorker, &CaptureWorker::fpsUpdated, this, &VideoController::updateFps);
    connect(m_inferenceWorker, &InferenceWorker::detectionsReady, this, &VideoController::updateDetections);

    // System Monitor
    m_systemMonitor = new SystemMonitor(this);
    connect(m_systemMonitor, &SystemMonitor::resourceUsageUpdated, this, &VideoController::updateSystemStats);

    // Start Threads
    m_inferenceThread.start(); // This triggers startInference()
    m_captureThread.start();
}

VideoController::~VideoController() {
    emit stopWorkers();
    m_captureThread.quit();
    m_captureThread.wait();
    
    m_inferenceThread.quit();
    m_inferenceThread.wait();

    delete m_captureWorker;
    delete m_inferenceWorker;
}

void VideoController::setVideoSink(QVideoSink* sink) {
    if (m_sink == sink) return;
    m_sink = sink;
    emit videoSinkChanged();

    if (m_sink) {
        emit startWorkers(m_sink);
        m_systemMonitor->startMonitoring();
    } else {
        emit stopWorkers();
        m_systemMonitor->stopMonitoring();
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
    QVariantList detectionsList;
    
    for (const auto& res : results) {
        QVariantMap det;
        det["classId"] = res.classId;
        det["confidence"] = res.confidence;
        det["label"] = QString::fromStdString(classNames[res.classId]);
        
        // Normalize coordinates (assuming 640x480 input frame)
        // Note: The UI VideoOutput might stretch, so we send relative coordinates (0.0 - 1.0)
        // or absolute if we know the size. 
        // YOLO results are absolute pixels relative to the input image size.
        // We know input is 640x480.
        
        float x = res.box.x;
        float y = res.box.y;
        float w = res.box.width;
        float h = res.box.height;
        
        det["x"] = x / 640.0;
        det["y"] = y / 480.0; // Depending on how we resized/padded?
        // Wait, current logic in inference.cpp does a fit/resize. 
        // The results coming out of YOLO are scaled back to the original image size?
        // Let's check inference.cpp PostProcess logic.
        // It uses `resizeScales`.
        // So `res.box` IS in the coordinate space of the ORIGINAL 'rawFrame'.
        // rawFrame is fixed to 640x480 in capture settings.
        
        det["w"] = w / 640.0;
        det["h"] = h / 480.0;
        
        detectionsList.append(det); // Corrected: Using append instead of push_back
    }
    
    m_detections = detectionsList;
    emit detectionsChanged();

    // Update timing
    if (std::abs(m_preProcessTime - timing.preProcessTime) > 0.1 ||
        std::abs(m_inferenceTime - timing.inferenceTime) > 0.1 ||
        std::abs(m_postProcessTime - timing.postProcessTime) > 0.1) {
        
        m_preProcessTime = timing.preProcessTime;
        m_inferenceTime = timing.inferenceTime;
        m_postProcessTime = timing.postProcessTime;
        emit timingChanged();
    }

    // Calculate Inference FPS
    static auto lastTime = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    double diff = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastTime).count();
    
    if (diff > 0) {
        double currentFps = 1000.0 / diff;
        // Simple smoothing
        if (m_inferenceFps == 0) m_inferenceFps = currentFps;
        else m_inferenceFps = m_inferenceFps * 0.9 + currentFps * 0.1;
        
        emit inferenceFpsChanged();
    }
    lastTime = now;
}
