#include "CaptureWorker.h"
#include <QThread>
#include <QCoreApplication>
#include <QVideoFrameFormat>
#include <QColor>
#include <chrono>
#include <QDebug>
#include "../../shared/domain/UiLogger.h"

CaptureWorker::CaptureWorker(ICaptureSource *source, QObject *parent)
    : QObject(parent)
    , m_source(source)
{
}

CaptureWorker::~CaptureWorker()
{
    stopCapturing();
}

void CaptureWorker::startCapturing(QVideoSink* sink)
{
    if (m_running) return;
    m_running = true;
    m_sink = sink;

    SourceConfig config;
    {
        std::lock_guard<std::mutex> lock(m_configMutex);
        config = m_requestedConfig;
    }

    if (!openSource(config)) {
        UiLogger::ctrl("CaptureWorker: Initial source not ready (waiting for input).");
    } else {
        UiLogger::ctrl("CaptureWorker: Source opened, starting capture loop...");
    }

    int frames = 0;
    auto startTime = std::chrono::high_resolution_clock::now();

    while (m_running) {
        QCoreApplication::processEvents();
        auto now = std::chrono::high_resolution_clock::now();

        if (m_configUpdatePending.load()) {
            SourceConfig configUpdate;
            {
                std::lock_guard<std::mutex> lock(m_configMutex);
                configUpdate = m_requestedConfig;
            }
            openSource(configUpdate);
            m_configUpdatePending = false;
            
            // Sync local config for pacing logic
            config = configUpdate;

            frames = 0;
            startTime = std::chrono::high_resolution_clock::now();
        }

        // Sync logic for Video File mode
        if (config.sourceType == InputSourceType::VideoFile) {
            int64_t total = m_source->frameCount();
            
            if (m_isFirstFrame) {
                m_videoStartTime = std::chrono::high_resolution_clock::now();
                m_isFirstFrame = false;
            } else {
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_videoStartTime).count();
                double fps = m_source->nativeFps();
                int64_t expectedOffset = static_cast<int64_t>(elapsed * fps / 1000.0);
                int64_t expectedFrame = m_startFrameIndex + expectedOffset;
                
                // Skip frames if we are behind
                while (m_videoFramesRead < expectedFrame) {
                    if (!m_source->skipFrame()) break;
                    m_videoFramesRead++;
                    if (total > 0 && m_videoFramesRead >= total) break; 
                }
            }
        }

        cv::Mat& currentFrame = m_framePool[m_poolIndex];
        {
            std::lock_guard<std::mutex> lock(m_sourceMutex);
            if (!m_source || !m_source->readFrame(currentFrame) || currentFrame.empty()) {
                if (config.sourceType == InputSourceType::VideoFile && config.loop) {
                    m_source->seekToFrame(0);
                    m_videoStartTime = std::chrono::high_resolution_clock::now();
                    m_videoFramesRead = 0;
                    m_startFrameIndex = 0;
                }
                QThread::msleep(10);
                continue;
            }
            if (config.sourceType == InputSourceType::VideoFile) {
                m_videoFramesRead = m_source->currentFrameIndex();
                int64_t total = m_source->frameCount();
                if (total > 0 && m_videoFramesRead >= total) {
                    m_videoFramesRead = total;
                }
                emit progressUpdated(m_videoFramesRead);
            }
        }

        bool isImage = (config.sourceType == InputSourceType::ImageFile);
        if (m_inferenceProcessingFlag && !m_inferenceProcessingFlag->load(std::memory_order_relaxed)) {
            if (!isImage || m_needsStaticInference.load()) {
                auto shared = std::make_shared<cv::Mat>(currentFrame.clone());
                emit frameReady(shared);
                m_needsStaticInference = false;
            }
        }

        std::shared_ptr<std::vector<DetectionResult>> currentDetections;
        {
            std::lock_guard<std::mutex> lock(m_detectionsMutex);
            currentDetections = m_latestDetections;
        }

        if (currentDetections) {
            for (const auto& det : *currentDetections) {
                if (!det.boxMask.empty()) {
                    cv::Rect originalBox = det.box;
                    cv::Rect displayBox = originalBox & cv::Rect(0, 0, currentFrame.cols, currentFrame.rows);
                    if (displayBox.width > 0 && displayBox.height > 0) {
                        int dx = displayBox.x - originalBox.x;
                        int dy = displayBox.y - originalBox.y;
                        cv::Rect maskRoi(dx, dy, displayBox.width, displayBox.height);
                        maskRoi = maskRoi & cv::Rect(0, 0, det.boxMask.cols, det.boxMask.rows);
                        
                        if (maskRoi.width > 0 && maskRoi.height > 0) {
                            cv::Mat roi = currentFrame(displayBox);
                            int hue = (det.classId * 60) % 360;
                            QColor color = QColor::fromHsl(hue, 255, 127);
                            int b = color.blue();
                            int g = color.green();
                            int r = color.red();
                            
                            cv::Mat activeMask = det.boxMask(maskRoi);
                            if (activeMask.size() != roi.size()) {
                                cv::resize(activeMask, activeMask, roi.size());
                            }

                            for (int y = 0; y < roi.rows; ++y) {
                                uchar* pRoi = roi.ptr<uchar>(y);
                                const uchar* pMask = activeMask.ptr<uchar>(y);
                                for (int x = 0; x < roi.cols; ++x) {
                                    if (pMask[x] > 128) {
                                        pRoi[x*3+0] = (pRoi[x*3+0] + b) >> 1;
                                        pRoi[x*3+1] = (pRoi[x*3+1] + g) >> 1;
                                        pRoi[x*3+2] = (pRoi[x*3+2] + r) >> 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if (m_sink) {
            QVideoFrame& frame = m_reusableFrames[m_reusableFrameIndex];
            
            // If the frame is currently held by the sink, creating a new one is safer than failing
            if (!frame.map(QVideoFrame::WriteOnly)) {
                QVideoFrameFormat format(m_source->currentResolution(), QVideoFrameFormat::Format_BGRA8888);
                frame = QVideoFrame(format);
                frame.map(QVideoFrame::WriteOnly);
            }

            if (frame.isMapped()) {
                cv::Mat resizedFrame = currentFrame;
                if (currentFrame.cols != frame.width() || currentFrame.rows != frame.height()) {
                    cv::resize(currentFrame, resizedFrame, cv::Size(frame.width(), frame.height()));
                }

                // BGRA8888 in Qt matches OpenCV's BGRA
                cv::Mat wrapper(frame.height(), frame.width(), CV_8UC4, 
                              frame.bits(0), frame.bytesPerLine(0));
                              
                if (resizedFrame.channels() == 3) {
                    cv::cvtColor(resizedFrame, wrapper, cv::COLOR_BGR2BGRA);
                } else if (resizedFrame.channels() == 4) {
                    resizedFrame.copyTo(wrapper);
                } else if (resizedFrame.channels() == 1) {
                    cv::cvtColor(resizedFrame, wrapper, cv::COLOR_GRAY2BGRA);
                }
                
                frame.unmap();
                m_sink->setVideoFrame(frame);
            }
            m_reusableFrameIndex = (m_reusableFrameIndex + 1) % 4;
        }
        
        m_poolIndex = (m_poolIndex + 1) % 3;

        if (config.sourceType != InputSourceType::ImageFile) {
            frames++;
        } else {
            frames = 0; // Keep at 0 for static images
        }

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();
        if (duration >= 1000) {
            emit fpsUpdated(frames * 1000.0 / duration);
            frames = 0;
            startTime = now;
        }

        // Pacing for video files to maintain target FPS
        if (config.sourceType == InputSourceType::VideoFile) {
            double targetFps = m_source->nativeFps();
            if (targetFps > 0) {
                auto loopEnd = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(loopEnd - m_videoStartTime).count();
                // Frame offset since the last sync point (start or seek)
                int64_t framesSinceSync = m_videoFramesRead - m_startFrameIndex;
                int64_t nextFrameTime = static_cast<int64_t>(framesSinceSync * 1000.0 / targetFps);
                
                if (elapsed < nextFrameTime) {
                    QThread::msleep(nextFrameTime - elapsed);
                }
            }
        }
    }
    {
        std::lock_guard<std::mutex> lock(m_sourceMutex);
        if (m_source) m_source->close();
    }
    emit cleanUp();
}

bool CaptureWorker::openSource(const SourceConfig& config) {
    std::lock_guard<std::mutex> lock(m_sourceMutex);
    if (!m_source) {
        UiLogger::ctrl("CaptureWorker: Error - No source to open.");
        return false;
    }

    UiLogger::ctrl("CaptureWorker: Opening source (Mode=" + 
                   QString(config.sourceType == InputSourceType::LiveCamera ? "Camera" : "Video") + ")");

    if (!m_source->open(config)) {
        UiLogger::ctrl("CaptureWorker: Failed to open source.");
        return false;
    }

    QSize actual = m_source->currentResolution();
    
    if (config.sourceType == InputSourceType::VideoFile) {
        emit metadataUpdated(m_source->nativeFps(), m_source->frameCount());
    }

    QVideoFrameFormat format(actual, QVideoFrameFormat::Format_BGRA8888);
    for (int i = 0; i < 4; ++i) m_reusableFrames[i] = QVideoFrame(format);
    m_reusableFrameIndex = 0;

    for(int i=0; i<3; ++i) m_framePool[i] = cv::Mat();
    clearDetections();

    // Reset sync state
    m_videoStartTime = std::chrono::high_resolution_clock::now();
    m_videoFramesRead = 0;
    m_startFrameIndex = 0;
    m_isFirstFrame = true;
    m_needsStaticInference = true;
    
    emit resolutionChanged(actual);
    return true;
}

void CaptureWorker::stopCapturing() {
    m_running = false;
}

void CaptureWorker::requestSeek(int64_t frame) {
    UiLogger::ctrl(QString("CaptureWorker: Seek requested to frame %1").arg(frame));
    std::lock_guard<std::mutex> lock(m_sourceMutex);
    if (!m_source) return;
    
    if (m_source->seekToFrame(frame)) {
        // Sync m_videoFramesRead with ACTUAL frame index after seeking
        m_videoFramesRead = m_source->currentFrameIndex();
        m_startFrameIndex = m_videoFramesRead;
        m_videoStartTime = std::chrono::high_resolution_clock::now();
        
        UiLogger::ctrl(QString("CaptureWorker: Seek successful. Actual frame: %1").arg(m_videoFramesRead));
        emit progressUpdated(m_videoFramesRead);
    } else {
        UiLogger::ctrl("CaptureWorker: Seek failed.");
    }
}

void CaptureWorker::updateResolution(const QSize& size) {
    {
        std::lock_guard<std::mutex> lock(m_configMutex);
        m_requestedConfig.resolution = size;
    }
    m_configUpdatePending = true;
}

void CaptureWorker::setSource(ICaptureSource* source, const SourceConfig& config) {
    UiLogger::ctrl("CaptureWorker: setSource requested.");
    {
        std::lock_guard<std::mutex> configLock(m_configMutex);
        m_requestedConfig = config;
    }
    
    {
        std::lock_guard<std::mutex> sourceLock(m_sourceMutex);
        if (m_source) {
            UiLogger::ctrl("CaptureWorker: Closing old source.");
            m_source->close();
        }
        m_source = source;
    }
    
    m_configUpdatePending = true;
}

void CaptureWorker::updateLatestDetections(std::shared_ptr<std::vector<DetectionResult>> detections, const QSize& frameSize) {
    Q_UNUSED(frameSize);
    std::lock_guard<std::mutex> lock(m_detectionsMutex);
    m_latestDetections = detections;
}

void CaptureWorker::clearDetections() {
    std::lock_guard<std::mutex> lock(m_detectionsMutex);
    m_latestDetections.reset();
}
