#include "VideoFileController.h"
#include "../../shared/domain/UiLogger.h"

VideoFileController::VideoFileController(QObject *parent)
    : QObject(parent)
    , m_source(std::make_unique<OpenCVVideoFileSource>())
{
}

void VideoFileController::setFilePath(const QUrl& fileUrl) {
    QString path = fileUrl.toLocalFile();
    if (m_filePath != path) {
        m_filePath = path;
        UiLogger::ctrl("VideoFileController: File path set → " + m_filePath);
        emit filePathChanged();

        SourceConfig config;
        config.sourceType = InputSourceType::VideoFile;
        config.filePath = m_filePath;
        config.loop = true;

        emit sourceReadyRequested(m_source.get(), config);
    }
}
