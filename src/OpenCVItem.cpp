#include "OpenCVItem.h"
#include <QPainter>

OpenCVItem::OpenCVItem(QQuickItem *parent) : QQuickPaintedItem(parent) {
  m_capture.open(0);

  // Start the FPS measurement timer
  m_fpsTimer.start();

  connect(&m_timer, &QTimer::timeout, this, &OpenCVItem::updateFrame);

  // Try to run at ~60 FPS (16ms) instead of 30 FPS for smoother video
  m_timer.start(16);
}

OpenCVItem::~OpenCVItem() {
  if (m_capture.isOpened()) {
    m_capture.release();
  }
}

void OpenCVItem::updateFrame() {
  if (!m_capture.isOpened())
    return;

  cv::Mat rawFrame;
  m_capture >> rawFrame;
  if (rawFrame.empty())
    return;

  // --- FPS Calculation Start ---
  m_frameCount++;
  // If 1 second (1000ms) has passed
  if (m_fpsTimer.elapsed() >= 1000) {
    m_fps = m_frameCount; // Update the fps variable
    m_frameCount = 0;     // Reset counter
    m_fpsTimer.restart(); // Restart timer
    emit fpsChanged();    // Tell QML to update the text
  }
  // --- FPS Calculation End ---

  cv::Mat rgbFrame;
  cv::cvtColor(rawFrame, rgbFrame, cv::COLOR_BGR2RGB);

  m_currentFrame = QImage(rgbFrame.data, rgbFrame.cols, rgbFrame.rows,
                          rgbFrame.step, QImage::Format_RGB888)
                       .copy();
  update();
}

void OpenCVItem::paint(QPainter *painter) {
  if (m_currentFrame.isNull()) {
    painter->fillRect(boundingRect(), Qt::black);
    return;
  }
  painter->drawImage(boundingRect(), m_currentFrame);
}
