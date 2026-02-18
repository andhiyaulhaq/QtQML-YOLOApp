#ifndef VIDEOCONTROLLER_H
#define VIDEOCONTROLLER_H

#include "inference.h"
#include "SystemMonitor.h"
#include <QMutex>
#include <atomic>
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

signals:
    void fpsUpdated(double fps);

public slots:
  void startCapturing(QVideoSink *sink);
  void stopCapturing();

private:
  std::atomic<bool> m_running{false};
  cv::VideoCapture m_capture;
  YOLO_V8 *yolo = nullptr;
  std::vector<cv::Scalar> classColors;
  void initializeClassColors();
};

// --- The Controller Class (Connects to QML) ---
class VideoController : public QObject {
  Q_OBJECT
  QML_ELEMENT

  // VideoSink Property
  Q_PROPERTY(QVideoSink *videoSink READ videoSink WRITE setVideoSink NOTIFY
                 videoSinkChanged)
                 
  // Performance Monitoring Properties
  Q_PROPERTY(double fps READ fps NOTIFY fpsChanged)
  Q_PROPERTY(QString systemStats READ systemStats NOTIFY systemStatsChanged)

public:
  explicit VideoController(QObject *parent = nullptr);
  ~VideoController();

  QVideoSink *videoSink() const { return m_sink; }
  void setVideoSink(QVideoSink *sink);

  double fps() const { return m_fps; }
  QString systemStats() const { return m_systemStats; }

public slots:
    void updateFps(double fps);
    void updateSystemStats(const QString &cpu, const QString &sysMem, const QString &procMem);

signals:
  void videoSinkChanged();
  void fpsChanged();
  void systemStatsChanged();
  void startWorker(QVideoSink *sink);
  void stopWorker();

private:
  QVideoSink *m_sink = nullptr;
  QThread m_workerThread;
  CameraWorker *m_worker = nullptr;
  SystemMonitor *m_systemMonitor;
  
  double m_fps = 0.0;
  QString m_systemStats;
};

#endif // VIDEOCONTROLLER_H
