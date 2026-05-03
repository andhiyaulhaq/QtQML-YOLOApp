#include "ImageFileController.h"
#include "../../shared/domain/UiLogger.h"

ImageFileController::ImageFileController(QObject *parent)
    : QObject(parent)
    , m_source(std::make_unique<OpenCVImageFileSource>())
{
}

void ImageFileController::setFilePath(const QUrl& fileUrl) {
    QString path = fileUrl.toLocalFile();
    if (path.isEmpty()) return;

    bool changed = (m_filePath != path);
    m_filePath = path;
    
    if (changed) {
        UiLogger::ctrl("ImageFileController: Image path set → " + m_filePath);
        emit filePathChanged();
    } else {
        UiLogger::ctrl("ImageFileController: Re-selecting same image → " + m_filePath);
    }

    activate();
}

void ImageFileController::activate() {
    if (m_filePath.isEmpty()) {
        UiLogger::ctrl("ImageFileController: Cannot activate - No file path set.");
        return;
    }

    UiLogger::ctrl("ImageFileController: Activating image file source.");
    
    SourceConfig config;
    config.sourceType = InputSourceType::ImageFile;
    config.filePath = m_filePath;

    emit sourceReadyRequested(m_source.get(), config);
}
