#include "OpenCVItem.h"
#include <QPainter>

OpenCVItem::OpenCVItem(QQuickItem *parent) : QQuickPaintedItem(parent) {
  // Open default camera (ID 0)
  m_capture.open(0);

  // Optional: Set resolution to 640x480 for speed
  // m_capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  // m_capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

  // Setup a timer to trigger updates ~30 FPS (33ms)
  connect(&m_timer, &QTimer::timeout, this, &OpenCVItem::updateFrame);
  m_timer.start(33);
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

  // Convert BGR (OpenCV) to RGB (Qt)
  // We clone data because QImage uses existing memory, which cv::Mat might
  // delete
  cv::Mat rgbFrame;
  cv::cvtColor(rawFrame, rgbFrame, cv::COLOR_BGR2RGB);

  // Create QImage from the Mat data
  m_currentFrame =
      QImage(rgbFrame.data, rgbFrame.cols, rgbFrame.rows, rgbFrame.step,
             QImage::Format_RGB888)
          .copy(); // .copy() ensures deep copy

  // Trigger a repaint of the item
  update();
}

void OpenCVItem::paint(QPainter *painter) {
  if (m_currentFrame.isNull()) {
    painter->fillRect(boundingRect(), Qt::black);
    return;
  }

  // Draw the image scaled to fit the QML item size
  QRectF targetRect = boundingRect();
  painter->drawImage(targetRect, m_currentFrame);
}
