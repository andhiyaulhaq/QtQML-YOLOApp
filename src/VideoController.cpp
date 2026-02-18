#include "VideoController.h"
#include <fstream>
#include <thread>
#include <chrono>

// =========================================================
// WORKER IMPLEMENTATION (Background Thread)
// =========================================================

void CameraWorker::initializeClassColors() {
  // Generate consistent colors for each class using a predefined palette
  // for better visual distinction and consistency
  const int numClasses = 80; // COCO dataset has 80 classes
  classColors.resize(numClasses);

  // Create visually distinct colors for common classes
  for (int i = 0; i < numClasses; ++i) {
    // Use different color regions for different class types
    cv::Scalar color;

    if (i == 0) { // person - blue
      color = cv::Scalar(255, 0, 0);
    } else if (i >= 1 && i <= 7) { // vehicles - red tones
      color = cv::Scalar(0, 0, 200 + i * 8);
    } else if (i >= 14 && i <= 23) { // animals - green tones
      color = cv::Scalar(0, 200 + (i - 14) * 5, 0);
    } else if (i >= 24 && i <= 39) { // objects - orange/yellow tones
      color = cv::Scalar(0, 100 + (i - 24) * 5, 200 + (i - 24) * 3);
    } else if (i >= 57 && i <= 61) { // furniture - purple tones
      color = cv::Scalar(150 + (i - 57) * 10, 0, 150 + (i - 57) * 10);
    } else { // other classes - cyan/magenta tones
      float hue =
          fmodf(i * 137.5f, 360.0f); // Golden angle for better distribution
      color = cv::Scalar(
          static_cast<int>(128 + 127 * sinf(hue * 3.14159f / 180.0f)),
          static_cast<int>(128 +
                           127 * sinf((hue + 120.0f) * 3.14159f / 180.0f)),
          static_cast<int>(128 +
                           127 * sinf((hue + 240.0f) * 3.14159f / 180.0f)));
    }

    classColors[i] = color;
  }
}

void CameraWorker::startCapturing(QVideoSink *sink) {
  if (m_running)
    return;
  m_running = true;

  // Initialize class colors
  initializeClassColors();

  // Open Camera (Try DSHOW for better Windows performance)
  m_capture.open(0, cv::CAP_DSHOW);
  if (!m_capture.isOpened()) {
    m_capture.open(0);
  }

  // Optimization Settings
  m_capture.set(cv::CAP_PROP_FOURCC,
                cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
  m_capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  m_capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  m_capture.set(cv::CAP_PROP_FPS, 30);

  // Initialize YOLO with multi-threading optimizations
  yolo = new YOLO_V8;
  // Load classes
  std::ifstream file("inference/classes.txt");
  std::string line;
  while (std::getline(file, line)) {
    yolo->classes.push_back(line);
  }
  file.close();
  
  // Create session with optimized multi-threading configuration
  DL_INIT_PARAM params;
  params.modelPath = "inference/yolov8n.onnx";
  params.modelType = YOLO_DETECT_V8;
  params.imgSize = {640, 640};
  params.cudaEnable = false;
  params.intraOpNumThreads = std::thread::hardware_concurrency(); // Use all logical cores (8)
  params.interOpNumThreads = 1; // Keep low to avoid thread contention
  params.sessionPoolSize = 1; // Phase 1: Single session with multi-threading
  yolo->CreateSession(params);

  cv::Mat rawFrame; // No displayFrame needed for zero-copy

  // FPS Calculation Variables
  int frames = 0;
  auto startTime = std::chrono::high_resolution_clock::now();

  while (m_running) {
    if (!m_capture.isOpened()) {
      QThread::msleep(100);
      continue;
    }

    m_capture >> rawFrame;
    if (rawFrame.empty())
      continue;

    // Optimization: Removed redundant clone. RunSession treats input as const.
    // Drawing happens on rawFrame *after* inference is done.
    
    std::vector<DL_RESULT> results;
    yolo->RunSession(rawFrame, results);

    // Draw detections with consistent colors directly on rawFrame
    for (auto &re : results) {
      cv::Scalar color =
          (re.classId < classColors.size())
              ? classColors[re.classId]
              : cv::Scalar(255, 255, 255); // Default white for unknown classes
      cv::rectangle(rawFrame, re.box, color, 3);

      float confidence = floor(100 * re.confidence) / 100;
      std::string label = yolo->classes[re.classId] + " " +
                          std::to_string(confidence)
                              .substr(0, std::to_string(confidence).size() - 4);

      cv::rectangle(rawFrame, cv::Point(re.box.x, re.box.y - 25),
                    cv::Point(re.box.x + label.length() * 10, re.box.y), color,
                    cv::FILLED);

      cv::putText(rawFrame, label, cv::Point(re.box.x, re.box.y - 5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }

    // --- ZERO-COPY RENDERING OPTIMIZATION ---
    if (sink) {
        QVideoFrameFormat format(QSize(rawFrame.cols, rawFrame.rows),
                               QVideoFrameFormat::Format_RGBA8888);
        QVideoFrame frame(format);

        if (frame.map(QVideoFrame::WriteOnly)) {
            // Create a cv::Mat wrapper around the QVideoFrame's memory
            // using the frame's specific stride (bytesPerLine)
            cv::Mat wrapper(rawFrame.rows, rawFrame.cols, CV_8UC4, 
                          frame.bits(0), frame.bytesPerLine(0));
            
            // Direct conversion from BGR (OpenCV) to RGBA (Qt) into the mapped memory
            cv::cvtColor(rawFrame, wrapper, cv::COLOR_BGR2RGBA);
            
            frame.unmap();
            sink->setVideoFrame(frame);
        }
    }

    // --- FPS CALCULATION ---
    frames++;
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();
    
    if (duration >= 500) { // Update every 500ms
        double fps = frames * 1000.0 / duration;
        emit fpsUpdated(fps);
        frames = 0;
        startTime = now;
    }
  }

  m_capture.release();
}

void CameraWorker::stopCapturing() {
  m_running = false;
  if (yolo) {
    delete yolo;
    yolo = nullptr;
  }
}

// =========================================================
// CONTROLLER IMPLEMENTATION (Main UI Thread)
// =========================================================
VideoController::VideoController(QObject *parent) : QObject(parent) {
  m_worker = new CameraWorker();
  m_worker->moveToThread(&m_workerThread);

  // Initialize System Monitor
  m_systemMonitor = new SystemMonitor(this);
  connect(m_systemMonitor, &SystemMonitor::resourceUsageUpdated, 
          this, &VideoController::updateSystemStats);

  connect(this, &VideoController::startWorker, m_worker,
          &CameraWorker::startCapturing);
  connect(this, &VideoController::stopWorker, m_worker,
          &CameraWorker::stopCapturing, Qt::DirectConnection);
          
  // Connect FPS signal
  connect(m_worker, &CameraWorker::fpsUpdated, this, &VideoController::updateFps);

  m_workerThread.start();
}

void VideoController::updateFps(double fps) {
    if (qAbs(m_fps - fps) > 0.1) {
        m_fps = fps;
        emit fpsChanged();
    }
}

void VideoController::updateSystemStats(const QString &cpu, const QString &sysMem, const QString &procMem) {
    // Format stats for display
    QString stats = QString("CPU: %1 | RAM: %2").arg(cpu, procMem);
    if (m_systemStats != stats) {
        m_systemStats = stats;
        emit systemStatsChanged();
    }
}

VideoController::~VideoController() {
  emit stopWorker();
  m_workerThread.quit();
  m_workerThread.wait();
  delete m_worker;
}

void VideoController::setVideoSink(QVideoSink *sink) {
  if (m_sink == sink)
    return;
  m_sink = sink;
  emit videoSinkChanged();

  if (m_sink) {
    // Start system monitoring when camera starts
    m_systemMonitor->startMonitoring();
    emit startWorker(m_sink);
  } else {
    // Stop system monitoring when camera stops
    m_systemMonitor->stopMonitoring();
  }
}
