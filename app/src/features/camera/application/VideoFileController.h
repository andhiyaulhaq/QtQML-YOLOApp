#pragma once

#include <QObject>
#include <QUrl>
#include <memory>
#include "../domain/ICaptureSource.h"
#include "../infrastructure/OpenCVVideoFileSource.h"

class VideoFileController : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString filePath READ filePath NOTIFY filePathChanged)
    Q_PROPERTY(bool hasFile READ hasFile NOTIFY filePathChanged)

public:
    explicit VideoFileController(QObject *parent = nullptr);

    Q_INVOKABLE void setFilePath(const QUrl& fileUrl);
    Q_INVOKABLE void activate();

    QString filePath() const { return m_filePath; }
    bool hasFile() const { return !m_filePath.isEmpty(); }

signals:
    void filePathChanged();
    void sourceReadyRequested(ICaptureSource* source, SourceConfig config);

private:
    QString m_filePath;
    std::unique_ptr<OpenCVVideoFileSource> m_source;
};
