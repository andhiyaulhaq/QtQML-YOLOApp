#include "OpenCVVideoFileSource.h"
#include <QDebug>

OpenCVVideoFileSource::~OpenCVVideoFileSource() {
    close();
}

bool OpenCVVideoFileSource::open(const SourceConfig& config) {
    if (config.sourceType != InputSourceType::VideoFile) return false;
    close();
    
    m_filePath = config.filePath;
    m_loop = config.loop;

    if (m_filePath.isEmpty()) {
        qDebug() << "[OpenCVVideoFileSource]: Error - Empty file path.";
        return false;
    }

    m_capture.open(m_filePath.toStdString());

    if (!m_capture.isOpened()) {
        qDebug() << "[OpenCVVideoFileSource]: Error - Could not open video file:" << m_filePath;
        return false;
    }

    int w = static_cast<int>(m_capture.get(cv::CAP_PROP_FRAME_WIDTH));
    int h = static_cast<int>(m_capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    m_currentResolution = QSize(w, h);
    
    m_nativeFps = m_capture.get(cv::CAP_PROP_FPS);
    if (m_nativeFps <= 0) m_nativeFps = 30.0;

    qDebug() << "[OpenCVVideoFileSource]: Opened" << m_filePath 
             << "Res:" << m_currentResolution << "FPS:" << m_nativeFps;

    return true;
}

void OpenCVVideoFileSource::close() {
    if (m_capture.isOpened()) {
        m_capture.release();
    }
}

bool OpenCVVideoFileSource::readFrame(cv::Mat& outFrame) {
    if (!m_capture.isOpened()) return false;

    if (!m_capture.read(outFrame) || outFrame.empty()) {
        if (m_loop) {
            m_capture.set(cv::CAP_PROP_POS_FRAMES, 0);
            return m_capture.read(outFrame);
        }
        return false;
    }

    return true;
}

bool OpenCVVideoFileSource::skipFrame() {
    if (!m_capture.isOpened()) return false;
    
    if (!m_capture.grab()) {
        if (m_loop) {
            m_capture.set(cv::CAP_PROP_POS_FRAMES, 0);
            return m_capture.grab();
        }
        return false;
    }
    return true;
}

QSize OpenCVVideoFileSource::currentResolution() const {
    return m_currentResolution;
}

double OpenCVVideoFileSource::nativeFps() const {
    return m_nativeFps;
}
