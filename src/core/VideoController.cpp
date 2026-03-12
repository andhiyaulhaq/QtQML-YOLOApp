#include "VideoController.h"
#include <QDebug>
#include <QThread>
#include <fstream>
#include <algorithm>

// =========================================================
// CAPTURE WORKER
// =========================================================

void CaptureWorker::updateLatestDetections(std::shared_ptr<std::vector<DL_RESULT>> detections) {
    std::lock_guard<std::mutex> lock(m_detectionsMutex);
    m_latestDetections = detections;
}

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

    // Initialize UI double-buffer
    QVideoFrameFormat format(QSize(AppConfig::FrameWidth, AppConfig::FrameHeight),
                             QVideoFrameFormat::Format_RGBA8888);
    m_reusableFrames[0] = QVideoFrame(format);
    m_reusableFrames[1] = QVideoFrame(format);

    // cv::Mat rawFrame; // Replaced by pool
    int frames = 0;
    auto startTime = std::chrono::high_resolution_clock::now();

    while (m_running) {
        cv::Mat& currentFrame = m_framePool[m_poolIndex];
        
        if (!m_capture.read(currentFrame) || currentFrame.empty()) {
            QThread::msleep(10);
            continue;
        }

        // 1. Send to Inference FIRST (un-blended frame)
        if (m_inferenceProcessingFlag && !m_inferenceProcessingFlag->load(std::memory_order_relaxed)) {
            auto shared = std::make_shared<cv::Mat>(currentFrame.clone());
            emit frameReady(shared);
        }

        // 2. Blend latest available segmentation masks
        std::shared_ptr<std::vector<DL_RESULT>> currentDetections;
        {
            std::lock_guard<std::mutex> lock(m_detectionsMutex);
            currentDetections = m_latestDetections;
        }

        if (currentDetections) {
            for (const auto& det : *currentDetections) {
                if (!det.boxMask.empty()) {
                    cv::Rect originalBox = det.box;
                    cv::Rect displayBox = originalBox & cv::Rect(0, 0, currentFrame.cols, currentFrame.rows);
                    if (displayBox.width > 0 && displayBox.height > 0) {
                        int dx = displayBox.x - originalBox.x;
                        int dy = displayBox.y - originalBox.y;
                        cv::Rect maskRoi(dx, dy, displayBox.width, displayBox.height);
                        maskRoi = maskRoi & cv::Rect(0, 0, det.boxMask.cols, det.boxMask.rows);
                        
                        if (maskRoi.width > 0 && maskRoi.height > 0) {
                            cv::Mat roi = currentFrame(displayBox);
                            int hue = (det.classId * 60) % 360;
                            QColor color = QColor::fromHsl(hue, 255, 127);
                            cv::Mat colorRoi(displayBox.size(), currentFrame.type(), cv::Scalar(color.blue(), color.green(), color.red()));
                            
                            cv::Mat blended;
                            cv::addWeighted(roi, 0.5, colorRoi, 0.5, 0.0, blended);
                            
                            cv::Mat activeMask = det.boxMask(maskRoi).clone();
                            if (activeMask.size() != roi.size()) {
                                cv::resize(activeMask, activeMask, roi.size());
                            }
                            blended.copyTo(roi, activeMask);
                        }
                    }
                }
            }
        }

        // 3. Send to UI (Zero-copy optimization)
        if (sink) {
            QVideoFrame& frame = m_reusableFrames[m_reusableFrameIndex];
            
            if (frame.map(QVideoFrame::WriteOnly)) {
                cv::Mat wrapper(currentFrame.rows, currentFrame.cols, CV_8UC4, 
                              frame.bits(0), frame.bytesPerLine(0));
                cv::cvtColor(currentFrame, wrapper, cv::COLOR_BGR2RGBA);
                frame.unmap();
                sink->setVideoFrame(frame);
            }
            m_reusableFrameIndex = (m_reusableFrameIndex + 1) % 2;
        }
        
        // Advance pool index
        m_poolIndex = (m_poolIndex + 1) % 3;

        // 4. FPS Calculation
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
}

void InferenceWorker::startInference() {
    changeModel(1); // Start with Object Detection by default
    // m_running is handled inside changeModel or set hereafter if successful
    if (m_pipeline) {
        m_running = true;
    }
}

void InferenceWorker::changeRuntime(int runtimeType) {
    int oldRuntime = m_currentRuntimeType;
    m_currentRuntimeType = runtimeType; 
    
    // If we have an existing pipeline, we need to reload it with the new runtime
    if (m_pipeline) {
        changeModel(m_currentTaskType);
    }
}

void InferenceWorker::changeModel(int taskType) {
    bool wasRunning = m_running;
    m_running = false; // Pause processing
    
    // Wait until current processing is done
    while (m_isProcessing.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    
    int oldRuntime = m_currentRuntimeType;
    DL_INIT_PARAM params;
    params.runtimeType = (m_currentRuntimeType == 0) ? RUNTIME_OPENVINO : RUNTIME_ONNXRUNTIME;

    auto getModelPath = [&](const std::string& baseName) {
        std::string path;
        if (params.runtimeType == RUNTIME_OPENVINO) {
            std::string xmlPath = "assets/openvino/" + baseName + ".xml";
            std::ifstream f(xmlPath.c_str());
            if (f.good()) {
                path = xmlPath;
            } else {
                emit errorOccurred("Model Not Found", QString("OpenVINO model not found: %1.xml").arg(QString::fromStdString(baseName)));
                return std::string("");
            }
        } else {
            path = "assets/onnx/" + baseName + ".onnx";
            std::ifstream f(path.c_str());
            if (!f.good()) {
                emit errorOccurred("Model Not Found", QString("ONNX model not found: %1.onnx").arg(QString::fromStdString(baseName)));
                return std::string("");
            }
        }
        return path;
    };

    if (taskType == 1) { // Object Detection
        params.modelPath = getModelPath("yolov8n");
    } else if (taskType == 2) { // Pose Estimation
        params.modelPath = getModelPath("yolov8n-pose");
    } else if (taskType == 3) { // Image Segmentation
        params.modelPath = getModelPath("yolov8n-seg");
    } else {
        emit errorOccurred("Unsupported Task", QString("Task type %1 is not supported.").arg(taskType));
    }

    if (params.modelPath.empty()) {
        m_currentRuntimeType = oldRuntime; // Revert runtime if changeModel fails
        m_running = wasRunning;
        return;
    }

    params.modelType = (taskType == 1) ? YOLO_DETECT : (taskType == 2) ? YOLO_POSE : YOLO_SEG;
    params.imgSize = {AppConfig::ModelWidth, AppConfig::ModelHeight};
    params.cudaEnable = false; 
    
    unsigned int threads = std::thread::hardware_concurrency() / 2;
    params.intraOpNumThreads = std::max(1u, std::min(4u, threads)); 
    params.interOpNumThreads = 1;

    // Create a temporary pipeline to avoid breaking the current one if initialization fails
    auto tempPipeline = std::make_unique<YoloPipeline>();
    
    // Load Classes
    std::ifstream file("assets/classes.txt");
    std::string line;
    while (std::getline(file, line)) {
        tempPipeline->classes.push_back(line);
    }
    file.close();

    try {
        const char* status = tempPipeline->CreateSession(params);
        if (status != RET_OK) {
            emit errorOccurred("Initialization Error", QString("[YoloPipeline]: %1").arg(status));
            m_running = wasRunning;
            // Revert runtime if this was triggered by changeRuntime
            return;
        }
        
        // Success! Atomic swap
        m_pipeline = std::move(tempPipeline);
        m_currentTaskType = taskType; // Update state ONLY on success
        m_currentRuntimeType = params.runtimeType == RUNTIME_OPENVINO ? 0 : 1;
        
        m_running = true; 
        emit modelLoaded(m_currentTaskType, m_currentRuntimeType);
    } catch (const std::exception& e) {
        qWarning() << "Failed to create session:" << e.what();
        emit errorOccurred("Critical Error", QString("Exception during session creation: %1").arg(e.what()));
        m_running = wasRunning;
    }
}

void InferenceWorker::stopInference() {
    m_running = false;
}

void InferenceWorker::processFrame(std::shared_ptr<cv::Mat> frame) {
    if (!m_running || !m_pipeline || !frame) return;

    // Drop frame if already processing
    bool expected = false;
    if (!m_isProcessing.compare_exchange_strong(expected, true)) {
        return; 
    }

    // Run Inference
    std::vector<DL_RESULT> results;
    YoloPipeline::InferenceTiming timing;
    // We can use the input const reference directly if RunSession accepts it.
    // If not, we might need a const_cast or better, fix RunSession to take const cv::Mat&
    // For now, let's assume we will fix RunSession.
    m_pipeline->RunSession(*frame, results, timing);

    // Emit results
    emit detectionsReady(results, &m_pipeline->getClassNames(), timing);
    emit latestDetectionsReady(std::make_shared<std::vector<DL_RESULT>>(results));

    m_isProcessing = false;
}

// =========================================================
// VIDEO CONTROLLER
// =========================================================

VideoController::VideoController(QObject *parent) : QObject(parent) {
    qRegisterMetaType<std::shared_ptr<cv::Mat>>("std::shared_ptr<cv::Mat>");
    qRegisterMetaType<const std::vector<std::string>*>("const std::vector<std::string>*");
    qRegisterMetaType<std::shared_ptr<std::vector<DL_RESULT>>>("std::shared_ptr<std::vector<DL_RESULT>>");
    
    // Initialize Model
    m_detections = new DetectionListModel(this);
    // 1. Create Workers
    auto captureWorker = std::make_unique<CaptureWorker>();
    m_captureWorker = captureWorker.get();
    
    auto inferenceWorker = std::make_unique<InferenceWorker>();
    m_inferenceWorker = inferenceWorker.get();
    
    // Link synchronization flags to avoid allocating and queuing frames during busy interference loop
    m_captureWorker->setInferenceProcessingFlag(m_inferenceWorker->getProcessingFlag());
    
    captureWorker.release(); // Qt takes ownership via moveToThread logic expectation
    inferenceWorker.release();

    m_captureWorker->moveToThread(&m_captureThread);
    m_inferenceWorker->moveToThread(&m_inferenceThread);

    // 2. Wiring

    // Main -> Workers
    connect(this, &VideoController::taskChangedBus, m_inferenceWorker, &InferenceWorker::changeModel, Qt::QueuedConnection);
    connect(this, &VideoController::runtimeChangedBus, m_inferenceWorker, &InferenceWorker::changeRuntime, Qt::QueuedConnection);
    connect(this, &VideoController::startWorkers, m_captureWorker, &CaptureWorker::startCapturing, Qt::QueuedConnection);
    connect(this, &VideoController::stopWorkers, m_captureWorker, &CaptureWorker::stopCapturing, Qt::DirectConnection); 
    connect(this, &VideoController::stopWorkers, m_inferenceWorker, &InferenceWorker::stopInference, Qt::DirectConnection); 
    connect(&m_inferenceThread, &QThread::started, m_inferenceWorker, &InferenceWorker::startInference);
    
    // Capture -> Inference
    connect(m_captureWorker, &CaptureWorker::frameReady, m_inferenceWorker, &InferenceWorker::processFrame, Qt::QueuedConnection);

    // Inference -> Capture (Latest Segmentation Masks)
    connect(m_inferenceWorker, &InferenceWorker::latestDetectionsReady, m_captureWorker, &CaptureWorker::updateLatestDetections, Qt::DirectConnection);

    // Workers -> Main
    connect(m_captureWorker, &CaptureWorker::fpsUpdated, this, &VideoController::updateFps);
    connect(m_inferenceWorker, &InferenceWorker::detectionsReady, this, &VideoController::updateDetections);
    connect(m_inferenceWorker, &InferenceWorker::errorOccurred, this, &VideoController::handleInferenceError);
    connect(m_inferenceWorker, &InferenceWorker::modelLoaded, this, &VideoController::handleModelLoaded);

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

void VideoController::setCurrentTask(TaskType task) {
    if (m_currentTask == task) return;
    m_currentTask = task;
    emit currentTaskChanged();
    emit taskChangedBus(static_cast<int>(m_currentTask));
}

void VideoController::setCurrentRuntime(RuntimeType runtime) {
    if (m_currentRuntime == runtime) return;
    m_currentRuntime = runtime;
    emit currentRuntimeChanged();
    emit runtimeChangedBus(static_cast<int>(m_currentRuntime));
}

void VideoController::updateFps(double fps) {
    if (std::abs(m_fps - fps) > 0.1) {
        m_fps = fps;
        emit fpsChanged();
    }
}

void VideoController::updateSystemStats(const QString &formattedStats) {
    if (m_systemStats != formattedStats) {
        m_systemStats = formattedStats;
        emit systemStatsChanged();
    }
}

void VideoController::updateDetections(const std::vector<DL_RESULT>& results, const std::vector<std::string>* classNames, const YoloPipeline::InferenceTiming& timing) {
    // Update the model directly
    if (m_detections && classNames) {
        m_detections->updateDetections(results, *classNames);
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

void VideoController::handleInferenceError(const QString& title, const QString& message) {
    // Revert selection if it changed
    if (m_currentTask != m_lastGoodTask) {
        m_currentTask = m_lastGoodTask;
        emit currentTaskChanged();
    }
    if (m_currentRuntime != m_lastGoodRuntime) {
        m_currentRuntime = m_lastGoodRuntime;
        emit currentRuntimeChanged();
    }
    
    // Forward original error signal to QML
    emit errorOccurred(title, message);
}

void VideoController::handleModelLoaded(int task, int runtime) {
    m_lastGoodTask = static_cast<TaskType>(task);
    m_lastGoodRuntime = static_cast<RuntimeType>(runtime);
}
