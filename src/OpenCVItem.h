#ifndef OPENCVITEM_H
#define OPENCVITEM_H

#include <QElapsedTimer> // <-- Add this include
#include <QImage>
#include <QQuickPaintedItem>
#include <QTimer>
#include <opencv2/opencv.hpp>

class OpenCVItem : public QQuickPaintedItem {
  Q_OBJECT
  QML_ELEMENT

  // This makes "fps" readable in QML as "camera.fps"
  Q_PROPERTY(int fps READ fps NOTIFY fpsChanged)

public:
  OpenCVItem(QQuickItem *parent = nullptr);
  ~OpenCVItem();

  void paint(QPainter *painter) override;

  // Getter for the property
  int fps() const { return m_fps; }

signals:
  // Signal to tell QML the value changed
  void fpsChanged();

private slots:
  void updateFrame();

private:
  cv::VideoCapture m_capture;
  QImage m_currentFrame;
  QTimer m_timer;

  // FPS counting variables
  QElapsedTimer m_fpsTimer;
  int m_frameCount = 0;
  int m_fps = 0;
};

#endif // OPENCVITEM_H
