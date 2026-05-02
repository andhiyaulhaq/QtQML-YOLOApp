#pragma once

#include <QObject>
#include <QQmlEngine>
#include <QVideoSink>
#include <QVariantList>
#include <QSize>
#include <memory>
#include "CaptureWorker.h"
#include "../infrastructure/OpenCVCameraSource.h"

class YoloCameraController : public QObject {
    Q_OBJECT

    Q_PROPERTY(QVideoSink* videoSink READ videoSink WRITE setVideoSink NOTIFY videoSinkChanged)
    Q_PROPERTY(double cameraFps READ cameraFps NOTIFY cameraFpsChanged)
    Q_PROPERTY(QVariantList supportedResolutions READ supportedResolutions NOTIFY supportedResolutionsChanged)
    Q_PROPERTY(QSize currentResolution READ currentResolution WRITE setCurrentResolution NOTIFY currentResolutionChanged)

public:
    explicit YoloCameraController(CaptureWorker *worker, QObject *parent = nullptr);

    QVideoSink* videoSink() const { return m_sink; }
    void setVideoSink(QVideoSink* sink);

    double cameraFps() const { return m_cameraFps; }
    QVariantList supportedResolutions() const { return m_supportedResolutions; }
    QSize currentResolution() const { return m_currentResolution; }

public slots:
    void setCurrentResolution(const QSize& size);
    void updateFps(double fps);
    void handleResolutionChanged(QSize size);
    void activate();

signals:
    void videoSinkChanged();
    void cameraFpsChanged();
    void supportedResolutionsChanged();
    void currentResolutionChanged();
    
    void startCapture(QVideoSink* sink);
    void stopCapture();
    void sourceReadyRequested(ICaptureSource* source, SourceConfig config);

private:
    CaptureWorker *m_worker;
    QVideoSink* m_sink = nullptr;
    double m_cameraFps = 0.0;
    QVariantList m_supportedResolutions;
    QSize m_currentResolution = QSize(640, 480);
    std::unique_ptr<OpenCVCameraSource> m_source;

    void refreshResolutions();
};
