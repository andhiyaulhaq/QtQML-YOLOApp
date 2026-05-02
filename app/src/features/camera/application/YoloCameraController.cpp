#include "YoloCameraController.h"
#include <QMediaDevices>
#include <QCameraDevice>
#include <QMetaObject>
#include "../../shared/domain/UiLogger.h"

YoloCameraController::YoloCameraController(CaptureWorker *worker, QObject *parent)
    : QObject(parent)
    , m_worker(worker)
{
    refreshResolutions();
    if (!m_supportedResolutions.isEmpty()) {
        m_currentResolution = m_supportedResolutions.last().toSize();
        m_worker->updateResolution(m_currentResolution);
    }
}

void YoloCameraController::setVideoSink(QVideoSink* sink)
{
    if (m_sink != sink) {
        m_sink = sink;
        emit videoSinkChanged();
        if (m_sink) {
            emit startCapture(m_sink);
        } else {
            emit stopCapture();
        }
    }
}

void YoloCameraController::setCurrentResolution(const QSize& size)
{
    UiLogger::ctrl("YoloCameraController::setCurrentResolution → " +
                   QString::number(size.width()) + "x" + QString::number(size.height()));
    if (m_currentResolution != size) {
        QMetaObject::invokeMethod(m_worker, "updateResolution", Qt::DirectConnection, Q_ARG(QSize, size));
    }
}

void YoloCameraController::updateFps(double fps)
{
    if (std::abs(m_cameraFps - fps) > 0.1) {
        m_cameraFps = fps;
        emit cameraFpsChanged();
    }
}

void YoloCameraController::handleResolutionChanged(QSize size)
{
    if (m_currentResolution != size) {
        m_currentResolution = size;
        emit currentResolutionChanged();
    }
}

void YoloCameraController::refreshResolutions()
{
    auto cameras = QMediaDevices::videoInputs();
    if (cameras.isEmpty()) return;

    auto formats = cameras.first().videoFormats();
    QSet<QString> unique;
    m_supportedResolutions.clear();

    for (const auto& format : formats) {
        QSize res = format.resolution();
        bool is480p = (res.width() == 640 && res.height() == 480);
        bool is720p = (res.width() == 1280 && res.height() == 720);
        
        if (!is480p && !is720p) continue;
        
        QString key = QString("%1x%2").arg(res.width()).arg(res.height());
        if (!unique.contains(key)) {
            unique.insert(key);
            m_supportedResolutions.append(res);
        }
    }

    std::sort(m_supportedResolutions.begin(), m_supportedResolutions.end(), [](const QVariant& a, const QVariant& b){
        return (a.toSize().width() * a.toSize().height()) < (b.toSize().width() * b.toSize().height());
    });

    emit supportedResolutionsChanged();
}
