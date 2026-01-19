#include "VideoController.h"

// =========================================================
// WORKER IMPLEMENTATION (Background Thread)
// =========================================================
void CameraWorker::startCapturing(QVideoSink *sink) {
  if (m_running)
    return;
  m_running = true;

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

  cv::Mat rawFrame, displayFrame;

  while (m_running) {
    if (!m_capture.isOpened()) {
      QThread::msleep(100);
      continue;
    }

    m_capture >> rawFrame;
    if (rawFrame.empty())
      continue;

    // Convert Color BGR -> RGBA
    cv::cvtColor(rawFrame, displayFrame, cv::COLOR_BGR2RGBA);

    // Send to VideoSink
    if (sink) {
      QVideoFrameFormat format(QSize(displayFrame.cols, displayFrame.rows),
                               QVideoFrameFormat::Format_RGBA8888);
      QVideoFrame frame(format);

      if (frame.map(QVideoFrame::WriteOnly)) {
        memcpy(frame.bits(0), displayFrame.data,
               displayFrame.total() * displayFrame.elemSize());
        frame.unmap();
        sink->setVideoFrame(frame);
      }
    }
  }

  m_capture.release();
}

void CameraWorker::stopCapturing() { m_running = false; }

// =========================================================
// CONTROLLER IMPLEMENTATION (Main UI Thread)
// =========================================================
VideoController::VideoController(QObject *parent) : QObject(parent) {
  m_worker = new CameraWorker();
  m_worker->moveToThread(&m_workerThread);

  connect(this, &VideoController::startWorker, m_worker,
          &CameraWorker::startCapturing);
  connect(this, &VideoController::stopWorker, m_worker,
          &CameraWorker::stopCapturing, Qt::DirectConnection);

  m_workerThread.start();
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
    emit startWorker(m_sink);
  }
}
