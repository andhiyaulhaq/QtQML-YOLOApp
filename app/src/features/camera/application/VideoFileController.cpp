#include "VideoFileController.h"
#include "../../shared/domain/UiLogger.h"

VideoFileController::VideoFileController(QObject *parent)
    : QObject(parent)
    , m_source(std::make_unique<OpenCVVideoFileSource>())
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

    emit sourceReadyRequested(m_source.get(), config);
}
