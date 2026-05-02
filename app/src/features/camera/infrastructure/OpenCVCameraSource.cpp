#include "OpenCVCameraSource.h"

OpenCVCameraSource::OpenCVCameraSource() {}

OpenCVCameraSource::~OpenCVCameraSource() {
    close();
}

bool OpenCVCameraSource::open(const CameraConfig& config) {
    close();
    
#ifdef _WIN32
    m_capture.open(config.deviceId, cv::CAP_MSMF);
#else
    m_capture.open(config.deviceId);
#endif

    if (!m_capture.isOpened()) {
        m_capture.open(config.deviceId);
    }

    if (!m_capture.isOpened()) return false;

    m_capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    m_capture.set(cv::CAP_PROP_FRAME_WIDTH, config.resolution.width());
    m_capture.set(cv::CAP_PROP_FRAME_HEIGHT, config.resolution.height());
    m_capture.set(cv::CAP_PROP_FPS, config.fps);

    int w = static_cast<int>(m_capture.get(cv::CAP_PROP_FRAME_WIDTH));
    int h = static_cast<int>(m_capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    m_currentResolution = QSize(w, h);

    return true;
}

void OpenCVCameraSource::close() {
    if (m_capture.isOpened()) {
        m_capture.release();
    }
}

bool OpenCVCameraSource::readFrame(cv::Mat& outFrame) {
    if (!m_capture.isOpened()) return false;
    return m_capture.read(outFrame);
}

QSize OpenCVCameraSource::currentResolution() const {
    return m_currentResolution;
}
