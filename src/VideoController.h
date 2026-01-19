#ifndef VIDEOCONTROLLER_H
#define VIDEOCONTROLLER_H

#include <QMutex>
#include <QObject>
#include <QQmlEngine>
#include <QThread>
#include <QVideoFrame>
#include <QVideoSink>
#include <QWaitCondition>
#include <opencv2/opencv.hpp>

// --- The Worker Class (Runs in Background) ---
class CameraWorker : public QObject {
  Q_OBJECT
public:
  explicit CameraWorker(QObject *parent = nullptr) : QObject(parent) {}

public slots:
  void startCapturing(QVideoSink *sink);
  void stopCapturing();

private:
  bool m_running = false;
  cv::VideoCapture m_capture;
};

// --- The Controller Class (Connects to QML) ---
class VideoController : public QObject {
  Q_OBJECT
  QML_ELEMENT

  // Only the VideoSink property remains
  Q_PROPERTY(QVideoSink *videoSink READ videoSink WRITE setVideoSink NOTIFY
                 videoSinkChanged)

public:
  explicit VideoController(QObject *parent = nullptr);
  ~VideoController();

  QVideoSink *videoSink() const { return m_sink; }
  void setVideoSink(QVideoSink *sink);

signals:
  void videoSinkChanged();
  void startWorker(QVideoSink *sink);
  void stopWorker();

private:
  QVideoSink *m_sink = nullptr;
  QThread m_workerThread;
  CameraWorker *m_worker = nullptr;
};

#endif // VIDEOCONTROLLER_H
