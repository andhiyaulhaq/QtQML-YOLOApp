#include "OpenCVImageFileSource.h"
#include <QDebug>
#include <QThread>

OpenCVImageFileSource::~OpenCVImageFileSource() {
    close();
}

bool OpenCVImageFileSource::open(const SourceConfig& config) {
    close();
    
    m_image = cv::imread(config.filePath.toStdString());
    if (m_image.empty()) {
        qDebug() << "[OpenCVImageFileSource]: Failed to load image from" << config.filePath;
        return false;
    }
    
    m_resolution = QSize(m_image.cols, m_image.rows);
    qDebug() << "[OpenCVImageFileSource]: Loaded image" << config.filePath << "at" << m_resolution;
    
    return true;
}

void OpenCVImageFileSource::close() {
    m_image.release();
    m_resolution = QSize(0, 0);
}

bool OpenCVImageFileSource::readFrame(cv::Mat& outFrame) {
    if (m_image.empty()) return false;
    
    // Simulate ~30fps to avoid 100% CPU usage for static images
    QThread::msleep(33);
    
    m_image.copyTo(outFrame);
    return true;
}

QSize OpenCVImageFileSource::currentResolution() const {
    return m_resolution;
}
