#include "CaptureWorker.h"
#include <QThread>
#include <QVideoFrameFormat>
#include <QColor>
#include <chrono>
#include <QDebug>

CaptureWorker::CaptureWorker(ICameraSource *source, QObject *parent)
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

    CameraConfig config;
    {
        std::lock_guard<std::mutex> lock(m_resolutionMutex);
        config.resolution = m_requestedResolution;
    }

    if (!openCamera(config)) {
        qDebug() << "CaptureWorker: Failed to open camera.";
        m_running = false;
        return;
    }

    qDebug() << "CaptureWorker: Camera opened, starting capture loop...";

    int frames = 0;
    auto startTime = std::chrono::high_resolution_clock::now();

    while (m_running) {
        if (m_resolutionUpdatePending.load()) {
            config.resolution = m_requestedResolution;
            openCamera(config);
            m_resolutionUpdatePending = false;
        }

        cv::Mat& currentFrame = m_framePool[m_poolIndex];
        if (!m_source->readFrame(currentFrame) || currentFrame.empty()) {
            QThread::msleep(10);
            continue;
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
                    cv::Rect displayBox = det.box & cv::Rect(0, 0, currentFrame.cols, currentFrame.rows);
                    if (displayBox.width > 0 && displayBox.height > 0) {
                        cv::Mat roi = currentFrame(displayBox);
                        int hue = (det.classId * 60) % 360;
                        QColor color = QColor::fromHsl(hue, 255, 127);
                        int b = color.blue();
                        int g = color.green();
                        int r = color.red();
                        
                        cv::Mat activeMask = det.boxMask;
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
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();
        if (duration >= 1000) {
            emit fpsUpdated(frames * 1000.0 / duration);
            frames = 0;
            startTime = now;
        }
    }
    m_source->close();
    emit cleanUp();
}

bool CaptureWorker::openCamera(const CameraConfig& config) {
    if (!m_source->open(config)) return false;

    QSize actual = m_source->currentResolution();
    QVideoFrameFormat format(actual, QVideoFrameFormat::Format_RGBA8888);
    m_reusableFrames[0] = QVideoFrame(format);
    m_reusableFrames[1] = QVideoFrame(format);

    for(int i=0; i<3; ++i) m_framePool[i] = cv::Mat();
    clearDetections();
    
    emit resolutionChanged(actual);
    return true;
}

void CaptureWorker::stopCapturing() {
    m_running = false;
}

void CaptureWorker::updateResolution(const QSize& size) {
    {
        std::lock_guard<std::mutex> lock(m_resolutionMutex);
        m_requestedResolution = size;
    }
    m_resolutionUpdatePending = true;
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
