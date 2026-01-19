#ifndef OPENCVITEM_H
#define OPENCVITEM_H

#include <QImage>
#include <QQuickPaintedItem>
#include <QTimer>
#include <opencv2/opencv.hpp>

class OpenCVItem : public QQuickPaintedItem {
  Q_OBJECT
  QML_ELEMENT // Registers this class to QML automatically

      public : OpenCVItem(QQuickItem *parent = nullptr);
  ~OpenCVItem();

  void paint(QPainter *painter) override;

private slots:
  void updateFrame();

private:
  cv::VideoCapture m_capture;
  QImage m_currentFrame;
  QTimer m_timer;
};

#endif // OPENCVITEM_H
