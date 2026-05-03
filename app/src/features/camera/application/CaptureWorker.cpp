#include "CaptureWorker.h"
#include <QThread>
#include <QCoreApplication>
#include <QVideoFrameFormat>
#include <QColor>
#include <chrono>
#include <QDebug>
#include "../../shared/domain/UiLogger.h"
#include "../infrastructure/OpenCVVideoFileSource.h"

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
            auto* fileSource = dynamic_cast<OpenCVVideoFileSource*>(m_source);
            if (fileSource) {
                int64_t total = fileSource->frameCount();
                
                if (m_isFirstFrame) {
                    m_videoStartTime = std::chrono::high_resolution_clock::now();
                    m_isFirstFrame = false;
                } else {
                    // Check for loop reset
                    if (total > 0 && m_videoFramesRead >= total) {
                        m_videoStartTime = std::chrono::high_resolution_clock::now();
                        m_videoFramesRead = 0;
                    }

                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_videoStartTime).count();
                    int64_t expectedFrame = static_cast<int64_t>(elapsed * fileSource->nativeFps() / 1000.0);
                    
                    // Skip frames if we are behind
                    while (m_videoFramesRead < expectedFrame) {
                        if (!fileSource->skipFrame()) break;
                        m_videoFramesRead++;
                        
                        // Prevent infinite loop if something goes wrong with frame counting
                        if (total > 0 && m_videoFramesRead >= total) break; 
                    }
                }
            }
        }

        cv::Mat& currentFrame = m_framePool[m_poolIndex];
        {
            std::lock_guard<std::mutex> lock(m_sourceMutex);
            if (!m_source || !m_source->readFrame(currentFrame) || currentFrame.empty()) {
                QThread::msleep(10);
                continue;
            }
            if (config.sourceType == InputSourceType::VideoFile) {
                m_videoFramesRead++;
                emit progressUpdated(m_videoFramesRead);
            }
        }

        if (m_inferenceProcessingFlag && !m_inferenceProcessingFlag->load(std::memory_order_relaxed)) {
            auto shared = std::make_shared<cv::Mat>(currentFrame.clone());
            emit frameReady(shared);
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
            if (frame.map(QVideoFrame::WriteOnly)) {
                cv::Mat resizedFrame = currentFrame;
                if (currentFrame.cols != frame.width() || currentFrame.rows != frame.height()) {
                    cv::resize(currentFrame, resizedFrame, cv::Size(frame.width(), frame.height()));
                }

                cv::Mat wrapper(frame.height(), frame.width(), CV_8UC4, 
                              frame.bits(0), frame.bytesPerLine(0));
                              
                if (resizedFrame.channels() == 3) {
                    cv::cvtColor(resizedFrame, wrapper, cv::COLOR_BGR2RGBA);
                } else if (resizedFrame.channels() == 4) {
                    cv::cvtColor(resizedFrame, wrapper, cv::COLOR_BGRA2RGBA);
                } else if (resizedFrame.channels() == 1) {
                    cv::cvtColor(resizedFrame, wrapper, cv::COLOR_GRAY2RGBA);
                }
                
                frame.unmap();
                m_sink->setVideoFrame(frame);
            }
            m_reusableFrameIndex = (m_reusableFrameIndex + 1) % 2;
        }
        
        m_poolIndex = (m_poolIndex + 1) % 3;

        frames++;
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();
        if (duration >= 1000) {
            emit fpsUpdated(frames * 1000.0 / duration);
            frames = 0;
            startTime = now;
        }

        // Pacing for video files to maintain target FPS
        if (config.sourceType == InputSourceType::VideoFile) {
            auto* fileSource = dynamic_cast<OpenCVVideoFileSource*>(m_source);
            if (fileSource) {
                double targetFps = fileSource->nativeFps();
                auto loopEnd = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(loopEnd - m_videoStartTime).count();
                int64_t nextFrameTime = static_cast<int64_t>((m_videoFramesRead) * 1000.0 / targetFps);
                
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
        auto* fileSource = dynamic_cast<OpenCVVideoFileSource*>(m_source);
        if (fileSource) {
            emit metadataUpdated(fileSource->nativeFps(), fileSource->frameCount());
        }
    }

    QVideoFrameFormat format(actual, QVideoFrameFormat::Format_RGBA8888);
    m_reusableFrames[0] = QVideoFrame(format);
    m_reusableFrames[1] = QVideoFrame(format);

    for(int i=0; i<3; ++i) m_framePool[i] = cv::Mat();
    clearDetections();

    // Reset sync state
    m_videoStartTime = std::chrono::high_resolution_clock::now();
    m_videoFramesRead = 0;
    m_isFirstFrame = true;
    
    emit resolutionChanged(actual);
    return true;
}

void CaptureWorker::stopCapturing() {
    m_running = false;
}

void CaptureWorker::requestSeek(int64_t frame) {
    std::lock_guard<std::mutex> lock(m_sourceMutex);
    if (!m_source) return;
    
    if (m_source->seekToFrame(frame)) {
        m_videoFramesRead = frame;
        
        // Adjust sync timer to new position
        auto* fileSource = dynamic_cast<OpenCVVideoFileSource*>(m_source);
        if (fileSource) {
            double fps = fileSource->nativeFps();
            auto now = std::chrono::high_resolution_clock::now();
            auto offset = std::chrono::milliseconds(static_cast<int64_t>(frame * 1000.0 / fps));
            m_videoStartTime = now - offset;
        }
        
        emit progressUpdated(m_videoFramesRead);
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
