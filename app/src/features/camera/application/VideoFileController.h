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
    Q_PROPERTY(double durationSeconds READ durationSeconds NOTIFY durationChanged)
    Q_PROPERTY(int64_t totalFrames READ totalFrames NOTIFY durationChanged)
    Q_PROPERTY(int64_t currentFrame READ currentFrame NOTIFY progressChanged)
    Q_PROPERTY(QString currentTimeStr READ currentTimeStr NOTIFY progressChanged)
    Q_PROPERTY(QString totalTimeStr READ totalTimeStr NOTIFY durationChanged)

public:
    explicit VideoFileController(QObject *parent = nullptr);

    Q_INVOKABLE void setFilePath(const QUrl& fileUrl);
    Q_INVOKABLE void activate();

    QString filePath() const { return m_filePath; }
    bool hasFile() const { return !m_filePath.isEmpty(); }
    
    double durationSeconds() const { return m_durationSeconds; }
    int64_t totalFrames() const { return m_totalFrames; }
    int64_t currentFrame() const { return m_currentFrame; }
    QString currentTimeStr() const;
    QString totalTimeStr() const;

    Q_INVOKABLE void seek(double position);
    void onProgressUpdated(int64_t frame);
    void onMetadataUpdated(double fps, int64_t totalFrames);

signals:
    void filePathChanged();
    void durationChanged();
    void progressChanged();
    void sourceReadyRequested(ICaptureSource* source, SourceConfig config);
    void requestSeek(int64_t frame);

private:
    QString formatTime(double seconds) const;

    QString m_filePath;
    double m_durationSeconds = 0;
    int64_t m_targetFrame = -1;
    bool m_isSeeking = false;
    int64_t m_totalFrames = 0;
    int64_t m_currentFrame = 0;
    double m_fps = 30.0;
};
