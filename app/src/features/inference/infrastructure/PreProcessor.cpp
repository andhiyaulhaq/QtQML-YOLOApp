#include "PreProcessor.h"
#include "SimdUtils.h"

ImagePreProcessor::ImagePreProcessor(YoloTask::TaskType taskType, const std::vector<int>& imgSize)
    : m_taskType(taskType), m_imgSize(imgSize) {}

LetterboxInfo ImagePreProcessor::preProcess(const cv::Mat &iImg, cv::Mat &oImg) {
    int target_h = m_imgSize.at(0);
    int target_w = m_imgSize.at(1);
    
    float r = std::min(target_w / (float)iImg.cols, target_h / (float)iImg.rows);
    int resized_w = static_cast<int>(iImg.cols * r);
    int resized_h = static_cast<int>(iImg.rows * r);
    
    m_info.scale = 1.0f / r;
    m_info.padW = (target_w - resized_w) / 2;
    m_info.padH = (target_h - resized_h) / 2;

    if (oImg.size() != cv::Size(target_w, target_h) || oImg.type() != CV_8UC3) {
        oImg.create(target_h, target_w, CV_8UC3);
    }
    oImg.setTo(cv::Scalar(114, 114, 114)); // YOLO padding color

    cv::Mat roi = oImg(cv::Rect(m_info.padW, m_info.padH, resized_w, resized_h));
    cv::resize(iImg, roi, cv::Size(resized_w, resized_h));
    
    return m_info;
}

void ImagePreProcessor::preProcessImageToBlob(const cv::Mat& iImg, float* blob_data) {
    simd::hwc_to_chw_bgr_to_rgb_sse41(iImg.data, blob_data, iImg.cols, iImg.rows, iImg.step);
}
