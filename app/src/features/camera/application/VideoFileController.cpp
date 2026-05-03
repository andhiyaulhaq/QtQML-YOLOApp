#include "VideoFileController.h"
#include "../../shared/domain/UiLogger.h"
#include <QTime>

VideoFileController::VideoFileController(QObject *parent)
    : QObject(parent)
{
}

void VideoFileController::setFilePath(const QUrl& fileUrl) {
    QString path = fileUrl.toLocalFile();
    if (path.isEmpty()) return;

    bool changed = (m_filePath != path);
    m_filePath = path;
    
    if (changed) {
        UiLogger::ctrl("VideoFileController: File path set → " + m_filePath);
        emit filePathChanged();
        
        // Reset playback info for new file
        m_currentFrame = 0;
        m_totalFrames = 0;
        m_durationSeconds = 0;
        emit progressChanged();
        emit durationChanged();
    } else {
        UiLogger::ctrl("VideoFileController: Re-selecting same file → " + m_filePath);
    }

    activate();
}

void VideoFileController::activate() {
    if (m_filePath.isEmpty()) {
        UiLogger::ctrl("VideoFileController: Cannot activate - No file path set.");
        return;
    }

    UiLogger::ctrl("VideoFileController: Activating video file source.");
    
    SourceConfig config;
    config.sourceType = InputSourceType::VideoFile;
    config.filePath = m_filePath;
    config.loop = true;

    // We don't hold the source here anymore, CaptureWorker does.
    // We pass a new instance or the factory should handle it.
    // For now, we create a temporary source just to pass its pointer if needed,
    // but the architecture implies AppController manages the transition.
    emit sourceReadyRequested(new OpenCVVideoFileSource(), config);
}

void VideoFileController::onMetadataUpdated(double fps, int64_t totalFrames) {
    m_fps = fps;
    m_totalFrames = totalFrames;
    m_durationSeconds = (fps > 0) ? (static_cast<double>(totalFrames) / fps) : 0;
    
    emit durationChanged();
}

void VideoFileController::onProgressUpdated(int64_t frame) {
    if (m_currentFrame != frame) {
        m_currentFrame = frame;
        emit progressChanged();
    }
}

void VideoFileController::seek(double position) {
    if (m_totalFrames <= 0) return;
    
    int64_t targetFrame = static_cast<int64_t>(position * m_totalFrames);
    emit requestSeek(targetFrame);
}

QString VideoFileController::currentTimeStr() const {
    double seconds = (m_fps > 0) ? (static_cast<double>(m_currentFrame) / m_fps) : 0;
    return formatTime(seconds);
}

QString VideoFileController::totalTimeStr() const {
    return formatTime(m_durationSeconds);
}

QString VideoFileController::formatTime(double seconds) const {
    int s = static_cast<int>(seconds);
    int m = s / 60;
    s = s % 60;
    return QString("%1:%2").arg(m, 2, 10, QChar('0')).arg(s, 2, 10, QChar('0'));
}
